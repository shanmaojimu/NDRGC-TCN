# utils/init.py
import torch
import torch.nn as nn

def glorot_weight_zero_bias(m: nn.Module):
    # Conv/Linear：仅当权重为二维或更高维时使用 Xavier
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        if getattr(m, 'weight', None) is not None and m.weight.dim() >= 2:
            nn.init.xavier_uniform_(m.weight)
        if getattr(m, 'bias', None) is not None:
            nn.init.zeros_(m.bias)

    # 归一化层：权重=1，偏置=0
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
        if getattr(m, 'weight', None) is not None:
            nn.init.ones_(m.weight)
        if getattr(m, 'bias', None) is not None:
            nn.init.zeros_(m.bias)

    # Embedding：常用均匀/正态也可，按你项目习惯
    elif isinstance(m, nn.Embedding):
        if getattr(m, 'weight', None) is not None and m.weight.dim() >= 2:
            nn.init.xavier_uniform_(m.weight)

    # 其他裸 Parameter（如果在子模块上直接挂了 nn.Parameter）
    else:
        for name, p in m.named_parameters(recurse=False):
            if p is None:
                continue
            # 跳过已经由上面的分支初始化的典型模块参数
            if p.dim() >= 2:
                # 尽量只对明显是“权重矩阵”的参数用 Xavier；否则可以选择跳过
                try:
                    nn.init.xavier_uniform_(p)
                except Exception:
                    pass
            else:
                # 1D 参数（如可学习门控/比例系数）不做 Xavier
                # 可按需：nn.init.ones_(p) 或 nn.init.zeros_(p)
                pass
