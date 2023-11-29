import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import torchio as tio

class MRIImage(Dataset):
    def __init__(self, images, targets, transform=None):
        self.transform = transform
        self.images = images
        self.targets = targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        dwi_path = self.images.values[index][0]
        adc_path = self.images.values[index][1]
        flair_path = self.images.values[index][2]
        target_path = self.targets.values[index]

        dwi_image = nib.load(dwi_path).get_fdata()

        original_images = np.zeros((4, dwi_image.shape[0], dwi_image.shape[1], dwi_image.shape[2]))
        original_images[0] = dwi_image
        original_images[1] = nib.load(adc_path).get_fdata()
        original_images[2] = nib.load(flair_path).get_fdata()
        original_images[3] = nib.load(target_path).get_fdata()
        if self.transform is not None:
            original_images = self.transform(original_images)
            original_images[0:3] = tio.ZNormalization()(original_images[0:3])
        mask = (original_images[3] > 0.5).astype(np.float32)

        return original_images[0:3].astype(np.float32), mask