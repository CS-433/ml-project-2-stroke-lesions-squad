import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

class MRIImages(Dataset):
    def __init__(self, image_paths, target_paths, transform=None):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transform = transform
        self.images = os.listdir(image_paths)
        self.targets = os.listdir(target_paths)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        dwi_path = os.path.join(self.image_paths, self.images[3 * index])
        adc_path = os.path.join(self.image_paths, self.images[3 * index + 1])
        flair_path = os.path.join(self.image_paths, self.images[3 * index + 1])
        target_path = os.path.join(self.target_paths, self.targets[index])

        dwi_image = nib.load(dwi_path).get_fdata()
        adc_image = nib.load(adc_path).get_fdata()
        flair_image = nib.load(flair_path).get_fdata()
        mask_image = nib.load(target_path).get_fdata()

        image = np.stack([dwi_image, adc_image, flair_image], axis=2)
        mask = np.stack([mask_image], axis=2)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        return image, mask