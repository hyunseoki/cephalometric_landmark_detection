import torch
from argmax import softargmax2d


def L2_loss_metric(pred, target, device='cpu'):
    pred = softargmax2d(pred, device=device)
    target = softargmax2d(target, device=device)
    loss = torch.mean(torch.pow((pred - target), 2))

    return loss


def mean_radial_error(pred, target, device='cpu'):
    pred = softargmax2d(pred, device=device)
    target = softargmax2d(target, device=device)

    loss = torch.sum(torch.pow(pred - target, 2), axis=-1)
    loss = torch.mean(torch.sqrt(loss))

    return loss


if __name__ == '__main__':
    # from dataset import(
    #     LandmarkDataset, 
    #     get_valid_transforms
    # )

    # train_dataset = LandmarkDataset(
    #     base_folder=r'C:\Users\bed1\src\cephalometric_landmark_detection\data\val',
    #     transforms=get_valid_transforms(),
    # )

    # train_data_loader = torch.utils.data.DataLoader(
    #         train_dataset,
    #         batch_size=2,
    #         shuffle=False,
    #     )

    # iterator = iter(train_data_loader)
    # sample = next(iterator)
    # print(sample['id'])
    # mask = sample['target']

    # coords = heatmap_to_landmark(mask)

    # print(mean_radial_error(pred=coords, target=(coords + 2)))
    # print(L2_loss(pred=coords, target=(coords + 2)))


    # import numpy as np
    
    pred = torch.tensor([[
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
        ]])

    true = torch.tensor([[
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
    ]]) 

    # pred = torch.tensor(pred, dtype=torch.float32)
    # true = torch.tensor(true, dtype=torch.float32)
    # loss = torch.nn.MSELoss()

    # print(f'MSELoss {loss(input=pred, target=true)}')
    # print(f'L2_loss {L2_loss(pred=pred, target=true)}') ## DACFL : 11.81
    # print(f'MRE {mean_radial_error(pred=pred, target=true)}')