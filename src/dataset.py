import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import os
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class LandmarkDataset(Dataset):
    def __init__(self, base_folder, transforms=None):
        self.base_folder = base_folder        
        self.image_fns = glob.glob(os.path.join(base_folder, 'image', '*.png'))
        self.heatmap_folders = [os.path.join(base_folder, 'heatmap', str(i)) for i in range(101, 111)]
        self.num_landmarks = len(self.heatmap_folders)
        self.transforms =  transforms

        assert len(self.image_fns) > 0
        for folder in self.heatmap_folders: assert os.path.isdir(folder)
        
    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_fns[idx]), cv2.COLOR_BGR2GRAY)
        mask = list()

        for mark_idx, folder in enumerate(self.heatmap_folders):
            mask_fn = os.path.join(folder, os.path.basename(self.image_fns[idx]))
            mask.append(cv2.cvtColor(cv2.imread(mask_fn), cv2.COLOR_BGR2GRAY))      

        if self.transforms != None:
            transformed = self.transforms(image=img, masks=mask)
            img, mask = transformed['image'], transformed['masks']

            img = img / 255.0
            mask = [mask_img/mask_img.max() for mask_img in mask]            
            mask = np.array(mask)

            if type(img) == torch.Tensor:
                mask = torch.tensor(mask, dtype=torch.float32)

        sample = dict()
        sample['id'] = os.path.basename(self.image_fns[mark_idx])
        sample['input'] = img
        sample['target'] = mask

        return sample


def get_train_transforms():
    return A.Compose(
        [
            A.Affine(
                scale=(0.9, 1.1),
                rotate=(25),
                translate_percent=(0.05, 0.05),
                cval=0,
                cval_mask=0,
                p=0.1,
            ),
            A.ColorJitter(p=0.1),
            ToTensorV2(p=1.0),
        ]
    )

def get_valid_transforms():
    return A.Compose(
        [
            ToTensorV2(p=1.0),
        ]
    )

def get_test_transforms():
    return A.Compose(
        [
            ToTensorV2(p=1.0),
        ]
    )


if __name__ == '__main__':
    dataset = LandmarkDataset(r'data/train', transforms=get_train_transforms())
    # dataset = LandmarkDataset(r'data/train', transforms=None)
    sample = dataset[3]
    img, mask = sample['input'], sample['target']
    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.imshow(img, 'gray')
    plt.axis('off')

    plt.figure(2)
    for idx in range(10):
        plt.subplot(2,5,idx+1)
        plt.axis('off')
        plt.imshow(mask[idx], 'gray')

    plt.show()