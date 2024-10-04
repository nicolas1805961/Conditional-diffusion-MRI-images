import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from glob import glob
import monai.transforms as T
from torchvision import transforms
import pickle

class CustomDataloaderTrain(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.path_list = np.array(glob(os.path.join(path, '*.npy')))
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        with open(os.path.join(os.path.dirname(path), 'out', 'nnUNet_preprocessed', 'Task032_Lib', 'splits_final.pkl'), 'rb') as f:
            data = pickle.load(f)[0]

        self.path_list = [x for x in self.path_list if os.path.basename(x).split('.')[0] in data['train']]

        #self.path_list = self.path_list[:10]


    def transform(self, examples):
        #images = [self.preprocess(image.convert("RGB")) for image in examples["image"]]
        images = [self.preprocess(image) for image in examples["image"]]
        return {"images": images}

    def augment(self, image, mask, pixel_transform, spatial_transform):
        
        padding_mask = image == 0

        data = {'image': image, 'mask': mask}
        pixel_transformed = pixel_transform(data)
        image = pixel_transformed['image'].as_tensor()
        mask = pixel_transformed['mask']
        image[padding_mask] = 0
        data = {'image': image, 'mask': mask}

        spatial_transformed = spatial_transform(data)
        image = spatial_transformed['image'].as_tensor()
        mask = spatial_transformed['mask']

        return image, mask

    def set_up_augmentation_pipeline(self):
        spatial_transform = [
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True), 
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.5, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.5, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.5, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                            ]

        pixel_transform = [
                                        T.RandRicianNoised(keys=['image'], prob=0.5, std=0.075),
                                        T.RandGibbsNoised(keys=['image'], prob=0.5, alpha=[0.45, 0.75]),
                                        T.RandAdjustContrastd(keys=['image'], prob=0.5, gamma=(0.7, 1.5), allow_missing_keys=True),
                                        T.RandGaussianNoised(keys=['image'], prob=0.5, std=0.04, allow_missing_keys=True),
                                        T.RandGaussianSharpend(keys=['image'], prob=0.5, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ]

        return pixel_transform, spatial_transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        batch_paths = self.path_list[idx]

        current_pixel_transform = np.random.choice(self.pixel_transform)
        current_spatial_transform = np.random.choice(self.spatial_transform)
            
        data = np.load(batch_paths)

        img = data[0]
        gt = data[1]

        depth_idx = np.random.choice(img.shape[-1])
        img = img[:, :, depth_idx][None]
        gt = gt[:, :, depth_idx][None]

        img, gt = self.augment(image=img, mask=gt, pixel_transform=current_pixel_transform, spatial_transform=current_spatial_transform)

        gt = torch.nn.functional.one_hot(gt[0].long(), num_classes=4).permute(2, 0, 1).contiguous().float()

        img = 2 * ((img - img.min()) / (img.max() - img.min() + 1e-8)) - 1

        img = img.float()

        #fig, ax = plt.subplots(1, 1)
        #ax.imshow(img[0], cmap='gray')
        #plt.show()

        return img, gt
    



class CustomDataloaderVal(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path, test=False):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.path_list = np.array(glob(os.path.join(path, '*.npy')))
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        if test:
            task_id = 'Task036_Lib'
        else:
            task_id = 'Task032_Lib'

        with open(os.path.join(os.path.dirname(path), 'out', 'nnUNet_preprocessed', task_id, 'splits_final.pkl'), 'rb') as f:
            data = pickle.load(f)[0]

        self.path_list = [x for x in self.path_list if os.path.basename(x).split('.')[0] in data['val']]

    def transform(self, examples):
        #images = [self.preprocess(image.convert("RGB")) for image in examples["image"]]
        images = [self.preprocess(image) for image in examples["image"]]
        return {"images": images}

    def augment(self, image, mask, pixel_transform, spatial_transform):
        
        padding_mask = image == 0

        data = {'image': image, 'mask': mask}
        pixel_transformed = pixel_transform(data)
        image = pixel_transformed['image']
        mask = pixel_transformed['mask']
        image[padding_mask] = 0
        data = {'image': image, 'mask': mask}

        spatial_transformed = spatial_transform(data)
        image = spatial_transformed['image']
        mask = spatial_transformed['mask']

        return image, mask

    def set_up_augmentation_pipeline(self):
        spatial_transform = [
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True), 
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.5, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.5, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.5, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                            ]

        pixel_transform = [
                                        T.RandRicianNoised(keys=['image'], prob=0.5, std=0.075),
                                        T.RandGibbsNoised(keys=['image'], prob=0.5, alpha=[0.45, 0.75]),
                                        T.RandAdjustContrastd(keys=['image'], prob=0.5, gamma=(0.7, 1.5), allow_missing_keys=True),
                                        T.RandGaussianNoised(keys=['image'], prob=0.5, std=0.04, allow_missing_keys=True),
                                        T.RandGaussianSharpend(keys=['image'], prob=0.5, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ]

        return pixel_transform, spatial_transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        batch_paths = self.path_list[idx]
            
        data = np.load(batch_paths)

        img = data[0]
        gt = data[1]

        depth_idx = np.random.choice(img.shape[-1])
        gt = gt[:, :, depth_idx][None]
        gt = torch.from_numpy(gt)

        gt = torch.nn.functional.one_hot(gt[0].long(), num_classes=4).permute(2, 0, 1).contiguous().float()

        #fig, ax = plt.subplots(1, 1)
        #ax.imshow(img[0], cmap='gray')
        #plt.show()

        return gt