


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import os
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision


class LandmarkDataset(Dataset):
    def __init__(self, base_folder, transforms=None):

        self.base_folder = base_folder        
        self.image_fns = glob.glob(os.path.join(base_folder, 'image', '*.png'))
        self.heatmap_folders = [os.path.join(base_folder, 'heatmap', str(i)) for i in range(101, 111)]
        self.num_landmarks = len(self.heatmap_folders)

        assert len(self.image_fns) > 0
        for folder in self.heatmap_folders: assert os.path.isdir(folder)
        
    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_fns[idx]), cv2.COLOR_BGR2GRAY)
        mask = np.zeros(shape=(self.num_landmarks, img.shape[0], img.shape[1]))

        for mark_idx, folder in enumerate(self.heatmap_folders):
            mask_fn = os.path.join(folder, os.path.basename(self.image_fns[idx]))
            mask[mark_idx] = cv2.cvtColor(cv2.imread(mask_fn), cv2.COLOR_BGR2GRAY)
        
        return img


def get_train_transforms():
    return A.Compose(
        [
            A.Resize(height=528, width=528),
            # A.Flip(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(15, interpolation=cv2.INTER_CUBIC),
            A.CoarseDropout(p=0.5, max_holes=16, max_height=50, max_width=50, min_holes=4, min_height=25, min_width=25),
            A.Normalize(),
            ToTensorV2(p=1.0),
        ]
    )

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=528, width=528),
            A.Normalize(),
            ToTensorV2(p=1.0),
        ]
    )

def get_test_transforms():
    return A.Compose(
        [
            A.Resize(height=528, width=528),
            A.Normalize(),
            ToTensorV2(p=1.0),
        ]
    )



if __name__ == '__main__':
    dataset = LandmarkDataset(r'C:\Users\bed1\src\cephalometric_landmark_detection\data\train')
    img = dataset[3]
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()