import torch
import numpy as np


def softargmax1d(input, beta=100):
    *_, n = input.shape
    input = torch.nn.functional.softmax(beta * input, dim=-1)
    indices = torch.linspace(0, 1, n)
    result = torch.sum((n - 1) * input * indices, dim=-1)
    return result


def softargmax2d(input, beta=100, device='cpu'):
    *_, h, w = input.shape

    input = input.reshape(*_, h * w)
    input = torch.nn.functional.softmax(beta * input, dim=-1)

    indices_c, indices_r = np.meshgrid(
        np.linspace(0, 1, w),
        np.linspace(0, 1, h),
        indexing='xy'
    )

    indices_r = torch.tensor(np.reshape(indices_r, (-1, h * w)), device=device)
    indices_c = torch.tensor(np.reshape(indices_c, (-1, h * w)), device=device)

    result_r = torch.sum((h - 1) * input * indices_r, dim=-1)
    result_c = torch.sum((w - 1) * input * indices_c, dim=-1)

    result = torch.stack([result_r, result_c], dim=-1)

    return result