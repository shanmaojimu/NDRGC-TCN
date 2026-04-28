import numpy as np
import torch
import random
import errno
import os
import sys
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset,Dataset
import pandas as pd
from utils.cutmix import CutMix
from utils.random_crop import RandomCrop
from utils.random_erasing import RandomErasing


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def set_save_path(father_path, args):
    father_path = os.path.join(father_path, '{}'.format(time.strftime("%m_%d_%H_%M")))
    mkdir(father_path)
    args.log_path = father_path
    args.model_path = father_path
    args.result_path = father_path
    args.spatial_adj_path = father_path
    args.time_adj_path = father_path
    args.tensorboard_path = father_path
    return args




EOS = 1e-10
def normalize(adj):
    adj = F.relu(adj)
    inv_sqrt_degree = 1. / (torch.sqrt(torch.sum(adj,dim=-1,keepdim=False)) + EOS)
    return inv_sqrt_degree[:,None]*adj*inv_sqrt_degree[None,:]




def save(checkpoints, save_path):
    torch.save(checkpoints, save_path)

import numpy as np
import torch
import cv2

class ActivationsAndGradients:
    """ Extract activations and gradients from target layers """
    def __init__(self, model, target_layers, reshape_transform=None):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []

        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation)
            )
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(self.save_gradient)
                )
            else:
                self.handles.append(
                    target_layer.register_backward_hook(self.save_gradient)
                )

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    """ GradCAM for 3D or 4D outputs """
    def __init__(self, model, target_layers, reshape_transform=None, use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform
        )

    @staticmethod
    def get_cam_weights(grads):
        # grads: (B,C,L) or (B,C,H,W)
        if grads.ndim == 4:
            return np.mean(grads, axis=(2,3), keepdims=True)
        elif grads.ndim == 3:
            return np.mean(grads, axis=2, keepdims=True)
        else:
            raise ValueError(f"Unsupported grads dimension: {grads.ndim}")

    @staticmethod
    def get_loss(output, target_category):
        # output: (B,C) logits
        loss = 0
        for i in range(len(target_category)):
            loss += output[i, target_category[i]]
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)  # sum over channels
        return cam

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        return np.float32(result)

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        # 对 3D 输出，target_size 可以不改变
        target_size = None

        cam_per_target_layer = []
        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    def __call__(self, input_tensor, target_category):
        if self.cuda:
            input_tensor = input_tensor.cuda() if isinstance(input_tensor, torch.Tensor) else input_tensor

        output = self.activations_and_grads(input_tensor)
        if isinstance(output, tuple):
            output = output[0]  # 取 logits
        target_list = [target_category] * input_tensor.size(0)

        self.model.zero_grad()
        loss = self.get_loss(output, target_list)
        loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()


def compute_activated(model, input_tensor, target_layer, target_category, use_cuda=False):
    """
    Args:
        model: PyTorch 模型
        input_tensor: 输入的 EEG tensor (B,C,L)
        target_layer: 目标提取层
        target_category: 分类目标 (int)
        use_cuda: 是否使用 GPU (True/False)
    Returns:
        numpy array: 激活图 (B,L)
    """
    grad_cam = GradCAM(model, [target_layer], use_cuda=use_cuda)
    cam = grad_cam(input_tensor, target_category)
    return cam