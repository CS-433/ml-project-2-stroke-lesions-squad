import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import torchio as tio

class MRIImage(Dataset):
    def __init__(self, images, targets, transform=None):
        """
        Create a dataset from a dataframe of images and targets.
        Parameters
        ----------
        images : A list of all the image paths of shape (3, n_images)
        targets : A list of all the target paths of shape (1, n_images)
        transform : The transform to apply to the images, defaults to None
        """
        self.transform = transform
        self.images = images
        self.targets = targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Get an image and its target from the dataset using an index.
        Parameters
        ----------
        index : The index of the image to get

        Returns : The image of shape (3, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH) and its target of shape (1, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH)
        -------

        """
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
            original_images[0:3] = normalize(original_images[0:3])

        mask = np.array([original_images[3] > 0.5]).astype(np.float32)

        return original_images[0:3], mask

def normalize(image):
    """
    Normalize a 3D image
    Parameters
    ----------
    image: an image of shape (BATCH_SIZE, 3, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH)

    Returns: the normalized image
    -------
    """
    eps = 1e-10
    min_value = np.min(image)
    max_value = np.max(image)
    norm_0_1 = (image - min_value) / (max_value - min_value + eps)

    return np.clip(2*norm_0_1 - 1, -1, 1)