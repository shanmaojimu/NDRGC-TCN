import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import sys

sys.path.append('../')

from utils.tools import normalize
from abc import abstractmethod
from math import sqrt
from utils.init import glorot_weight_zero_bias

EOS = 1e-10


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, inputs):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class NeuDIFRefineSpatialGraph(nn.Module):
    """
    NeuDIF-Refine: Learnable Diffusion-AntiDiffusion Adjacency Refinement

    Input : x (B, W, C, T)
    Output: x_out (B, C, W*T), edge_weight (C, C)

    Key Features:
      - Initial A0 comes from prior (layout/COH/PLV) or identity matrix
      - Pairwise distances are constructed from channel embeddings z
        (obtained via robust statistics) to serve as the gradient of the smoothness energy
      - Small MLP f_theta(A) learns "anti-diffusion" residual
      - Each step applies projection: non-negative / symmetric / (optional) PSD,
        normalization + HandMI-loop
      - (Optional) Gumbel-TopK differentiable sparsification
    """

    def __init__(self,
                 n_nodes=22,
                 adj=None,  # prior adjacency matrix
                 k=2,  # propagation power
                 spatial_GCN=True,
                 device=0,
                 # ====== NeuDIF Hyperparameters ======
                 steps=2,  # diffusion-anti-diffusion iteration steps (1-3 is usually enough)
                 step_size=0.2,  # η: diffusion step size
                 beta=0.5,  # β: anti-diffusion strength
                 d_embed=64,  # channel embedding dimension for z_i
                 use_psd_proj=False,  # whether to use PSD projection (recommended off for large C)
                 use_prior_blend=True,  # blend with prior at each step
                 prior_blend_alpha=0.2,  # blending coefficient α
                 use_gumbel_topk=True,  # differentiable Top-K sparsification
                 topk_ratio=0.25,  # Top-K ratio per row
                 temperature=1.0  # Gumbel temperature (can be annealed)
                 ):
        super().__init__()
        self.n_nodes = n_nodes
        self.k = k
        self.spatial_GCN = spatial_GCN
        self.device = device

        self.steps = steps
        self.step_size = step_size
        self.beta = nn.Parameter(torch.tensor(float(beta)), requires_grad=True)

        self.use_psd_proj = use_psd_proj
        self.use_prior_blend = use_prior_blend
        self.prior_blend_alpha = prior_blend_alpha

        self.use_gumbel_topk = use_gumbel_topk
        self.topk_ratio = topk_ratio
        self.temperature = nn.Parameter(torch.tensor(float(temperature)), requires_grad=False)

        # Prior adjacency
        if adj is None:
            prior = torch.eye(n_nodes)
        else:
            prior = adj.clone().detach()
            prior = (prior + prior.T) / 2
            prior.fill_diagonal_(1.0)
        self.register_buffer("A_prior", prior)

        # Channel robust statistics → embedding
        self.node_proj = nn.Linear(1, d_embed, bias=False)
        nn.init.xavier_uniform_(self.node_proj.weight)

        # Anti-diffusion f_theta: element-wise gated residual (stable and efficient)
        self.residual_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.GELU(),
            nn.Linear(16, 1),
            nn.Softplus()  # keep non-negative residual to avoid destructive oscillation
        )
        # Learnable scaling for early training stability
        self.gamma_res = nn.Parameter(torch.tensor(1.0))

    # ============== Utility Functions ==============
    @staticmethod
    def _normalize(A, eps=1e-6):
        D = torch.clamp(A.sum(-1), min=eps)
        D_inv_sqrt = torch.diag_embed(D.pow(-0.5))
        return D_inv_sqrt @ A @ D_inv_sqrt

    @staticmethod
    def _symmetrize(A):
        return 0.5 * (A + A.transpose(-1, -2))

    @staticmethod
    def _relu_eye(A):
        A = torch.relu(A)
        A = A + torch.eye(A.size(-1), device=A.device)
        return A

    def _psd_project(self, A):
        """Project to Positive Semi-Definite matrix via eigenvalue clipping"""
        evals, evecs = torch.linalg.eigh(self._symmetrize(A))
        evals = torch.clamp(evals, min=0.0)
        A_psd = (evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2))
        return self._symmetrize(A_psd)

    def _gumbel_topk_mask(self, scores, k):
        """Differentiable Top-K via Gumbel noise"""
        if k >= scores.size(-1):
            return torch.ones_like(scores)
        g = -torch.empty_like(scores).exponential_().log()  # Gumbel(0,1)
        y = (scores + g) / torch.clamp(self.temperature, min=1e-6)
        _, topk_idx = torch.topk(y, k, dim=-1)
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, topk_idx, 1.0)
        # Soft gate to keep gradients (currently hard TopK + noise)
        return mask + torch.sigmoid(y) * 0.0

    def _node_stats(self, x):
        """
        Extract robust per-channel statistics from (B,W,C,T) -> (B,W,C,1)
        Can be replaced with stronger features (log-var, bandpower, etc.)
        """
        mean = x.mean(dim=-1, keepdim=True)
        energy = (x ** 2).mean(dim=-1, keepdim=True)
        stat = torch.cat([mean, energy], dim=-1).mean(-1, keepdim=True)
        return stat

    def _pairwise_sqdist(self, Z):
        """
        Z: (C, d) -> D2: (C, C)
        Squared Euclidean distance, used as gradient term for smoothness energy
        """
        G = Z @ Z.t()
        sq = (Z ** 2).sum(dim=-1, keepdim=True)
        D2 = sq + sq.t() - 2.0 * G
        return torch.relu(D2)  # numerical stability

    # ============== Core: Construct A = NeuDIF-refine(x) ==============
    def build_A(self, x):
        """
        x: (B, W, C, T)
        returns: A (C, C)
        """
        B, W, C, T = x.shape
        dev = x.device

        # ---- 1) Channel Embedding Z ----
        stat = self._node_stats(x)  # (B,W,C,1)
        Z = torch.tanh(self.node_proj(stat))  # (B,W,C,d)
        Z = Z.mean(dim=1).mean(dim=0)  # (C,d)

        # ---- 2) Initial A0 ----
        A = self.A_prior.to(dev).clone()

        # ---- 3) NeuDIF Iteration ----
        D2 = self._pairwise_sqdist(Z) * 0.5  # gradient term (C,C)

        for _ in range(self.steps):
            # Diffusion step
            A = A - self.step_size * D2

            # Anti-diffusion step
            A_in = A.unsqueeze(-1)  # (C,C,1)
            delta = self.residual_mlp(A_in) * self.gamma_res
            A = A + self.beta * delta.squeeze(-1)

            # Projection: non-negative + symmetric + (optional) PSD
            A = self._relu_eye(self._symmetrize(A))
            if self.use_psd_proj:
                A = self._psd_project(A)
                A = self._relu_eye(A)

            # Optional prior blending for stability
            # if HandMI.use_prior_blend:
            #     A = (1.0 - HandMI.prior_blend_alpha) * A + HandMI.prior_blend_alpha * HandMI.A_prior.to(dev)

        # ---- 4) Differentiable Sparsification (Gumbel-TopK) ----
        if self.use_gumbel_topk and self.topk_ratio > 0:
            k_row = max(1, int(self.topk_ratio * C))
            mask = self._gumbel_topk_mask(A, k_row)
            A = A * mask
            A = self._relu_eye(self._symmetrize(A))

        # ---- 5) Normalization + Self-loop Enhancement ----
        A = self._normalize(A)
        A = A + torch.eye(C, device=dev)
        return A

    # ============== Forward Pass ==============
    def forward(self, x):
        """
        x: (B, W, C, T) -> (B, C, W*T), edge_weight (C, C)
        """
        if not self.spatial_GCN:
            edge_weight = torch.eye(self.n_nodes, device=x.device)
        else:
            edge_weight = self.build_A(x)

        # Reshape and spatial propagation (consistent with original implementation)
        x = x.permute(0, 2, 1, 3).contiguous().view(x.shape[0], self.n_nodes, -1)  # (B, C, W*T)
        for _ in range(self.k):
            x = torch.matmul(edge_weight.to(x.device), x)
        return x, edge_weight


class Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        nn.Module.__init__(self)
        self.conv = conv
        self.activation = activation
        if bn:
            self.conv.bias = None
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class NDRGC_TCN(BaseModel):
    def __init__(self,
                 Adj,
                 in_chans,
                 n_classes,
                 time_window_num,
                 spatial_GCN,
                 time_GCN,
                 k_spatial,
                 k_time,
                 dropout,
                 input_time_length=125,
                 out_chans=64,
                 kernel_size=63,
                 slide_window=8,
                 sampling_rate=250,
                 device=0,
                 ):
        super(NDRGC_TCN, self).__init__()

        self.__dict__.update(locals())
        del self.self

        self.device = device
        self.time_window_num = time_window_num

        # Spatial convolution (channel mixing)
        self.spatialconv = Conv(nn.Conv1d(in_chans, out_chans, kernel_size=1, bias=False, groups=1),
                                bn=nn.BatchNorm1d(out_chans), activation=None)

        # Multi-scale dilated TCN for temporal feature extraction
        self.timeconv = nn.ModuleList()
        self.layers = 4
        dilation_factors = [1, 2, 4, 8]
        for i in range(self.layers):
            dilation = dilation_factors[i]
            self.timeconv.append(Conv(
                nn.Conv1d(out_chans, out_chans, kernel_size=kernel_size,
                          stride=1, padding=(kernel_size - 1) // 2, dilation=dilation, bias=False),
                bn=nn.BatchNorm1d(out_chans), activation=None))

        self.downSampling = nn.AvgPool1d(int(sampling_rate // 2), int(sampling_rate // 2))
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(out_chans * (input_time_length * slide_window // (sampling_rate // 2)), n_classes)
        )

        # NeuDIF spatial graph refinement module
        self.ge = nn.Sequential(
            NeuDIFRefineSpatialGraph(
                n_nodes=in_chans, adj=Adj, k=k_spatial, spatial_GCN=spatial_GCN, device=device,
                steps=2, step_size=0.2, beta=0.5, d_embed=64,
                use_psd_proj=False, use_prior_blend=True, prior_blend_alpha=0.2,
                use_gumbel_topk=True, topk_ratio=0.25, temperature=1.0
            ),
        )

        self.apply(glorot_weight_zero_bias)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], self.time_window_num, -1).permute(0, 2, 1, 3)
        x, node_weights = self.ge(x)  # NeuDIF refined spatial graph convolution

        x = self.spatialconv(x)  # channel-wise linear mapping

        for i in range(len(self.timeconv)):  # multi-scale dilated TCN
            x = self.timeconv[i](x)

        x = F.gelu(x)
        x = x.contiguous().view(x.shape[0], x.shape[1], self.slide_window, -1)

        x = self.downSampling(x)
        x = self.dp(x)
        x = x.view(x.shape[0], -1)

        features_before_fc = x  # features before final FC layer

        x = self.fc(x)

        return x, features_before_fc, node_weights