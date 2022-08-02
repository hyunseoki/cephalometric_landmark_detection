import torch
from argmax import softargmax2d


def L1_loss(pred, target):
    loss = torch.mean(torch.abs(pred - target))
    return loss


def L2_loss(pred, target):
    loss = torch.mean(torch.pow((pred - target), 2))
    return loss

def AC_loss(pred, target, device='cpu'):
    def angle_matrix(coords):
        batch_size, num_landmark, _ = coords.shape

        A = torch.zeros((batch_size, num_landmark, num_landmark), dtype=torch.float32)
        rad = torch.atan2(coords[:, :, 0], coords[:, :, 1])
        for i in range(num_landmark):
            A[:, i, :] = rad - rad[:, i].reshape(batch_size, 1).expand_as(rad)

        return A

    def distance_matrix(coords):
        batch_size, num_landmark, _ = coords.shape
        D = torch.zeros((batch_size, num_landmark, num_landmark), dtype=torch.float32)
        for i in range(num_landmark):
            d = coords - coords[:, i, :].reshape(batch_size, 1, 2).expand_as(coords)
            D[:, i, :] = torch.norm(d.type(torch.float32), p=2, dim=2)

        return D

    pred = softargmax2d(pred, device=device)
    target = softargmax2d(target, device=device)

    A_pred = angle_matrix(pred)
    A_target = angle_matrix(target)
    D_pred = distance_matrix(pred)
    D_target = distance_matrix(target)

    loss = torch.log2(1 + torch.norm(A_target - A_pred, p=2)) + torch.log2(1 + torch.norm(D_target - D_pred, p=2))
    return loss


if __name__ == '__main__':
    from dataset import(
        LandmarkDataset, 
        get_valid_transforms
    )

    train_dataset = LandmarkDataset(
        base_folder=r'C:\Users\bed1\src\cephalometric_landmark_detection\data\val',
        transforms=get_valid_transforms(),
    )

    train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=False,
        )

    iterator = iter(train_data_loader)
    sample = next(iterator)
    print(sample['id'])
    mask = sample['target']

    print(AC_loss(pred=mask, target=mask))

    # print(mean_radial_error(pred=coords, target=(coords + 2)))
    # print(L2_loss(pred=coords, target=(coords + 2)))


    # import numpy as np

    # pred = torch.tensor(pred, dtype=torch.float32)
    # true = torch.tensor(true, dtype=torch.float32)
    # loss = torch.nn.MSELoss()

    # print(f'MSELoss {loss(input=pred, target=true)}')
    # print(f'L2_loss {L2_loss(pred=pred, target=true)}') ## DACFL : 11.81
    # print(f'MRE {mean_radial_error(pred=pred, target=true)}')