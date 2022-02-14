import torch


def L1_loss(pred, target):
    loss = torch.mean(torch.abs(pred - target))
    return loss

def L2_loss(pred, target):
    loss = torch.mean(torch.pow((pred - target), 2))
    return loss

def mean_radial_error(pred, target):
    loss = pred - target
    loss = torch.pow(loss, 2)
    loss = torch.sum(loss, axis=1)
    loss = torch.sqrt(loss)
    loss = torch.mean(loss)

    return loss
    
if __name__ == '__main__':
    import numpy as np
    
    pred = np.array([
        [385, 279], 
        [379, 466],
        [458, 422],
        [431, 197],
        [555, 461],
        [676, 505],
        [729, 510],
        [753, 485],
        [744, 502],
        [619, 235],
        ])

    true = np.array([
        [379, 280], 
        [375, 469],
        [460, 428],
        [423, 200],
        [564, 466],
        [680, 503],
        [729, 509],
        [751, 491],
        [745, 503],
        [620, 233],
    ]) 

    pred = torch.tensor(pred, dtype=torch.float32)
    true = torch.tensor(true, dtype=torch.float32)
    loss = torch.nn.MSELoss()

    print(f'MSELoss {loss(input=pred, target=true)}')
    print(f'L2_loss {L2_loss(pred=pred, target=true)}') ## DACFL : 11.81
    print(f'MRE {mean_radial_error(pred=pred, target=true)}')