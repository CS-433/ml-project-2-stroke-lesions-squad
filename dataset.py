import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import torchio as tio

class MRIImageFull(Dataset):
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
            #original_images[0:3] = histogram(original_images[0:3])
            original_images = self.transform(original_images)
            original_images[0:3] = normalize(original_images[0:3])

        mask = np.array([original_images[3] > 0.5]).astype(np.float32)

        return original_images[0:3], mask



class MRIImage_patched(Dataset):
    def __init__(self, images, targets, transform=None,patch_size = (64, 64, 64),num_patches = 2):
        """
        Create a dataset from a dataframe of images and targets.
        Parameters
        ----------
        images : A list of all the image paths of shape (3, n_images)
        targets : A list of all the target paths of shape (1, n_images)
        transform : The transform to apply to the images, defaults to None
        patch_size : default for a base dim of 128x128x128
        num_patches : default to 4
        """
        self.transform = transform
        self.images = images
        self.targets = targets
        self.patch_size = patch_size  # Each channel is 128x128x128, patches will be 64x64x64
        self.num_patches = num_patches

    def __len__(self):
        return len(self.images)* self.num_patches

    def __getitem__(self, index):
        """
        Get an image and its target from the dataset using an index.
        Parameters
        ----------
        index : The index of the image to get

        Returns : The image of shape (3, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH) and its target of shape (1, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH)
        -------

        """
        dwi_path = self.images.values[index//self.num_patches][0]
        adc_path = self.images.values[index//self.num_patches][1]
        flair_path = self.images.values[index//self.num_patches][2]
        target_path = self.targets.values[index//self.num_patches]

        dwi_image = nib.load(dwi_path).get_fdata()

        original_images = np.zeros((4, dwi_image.shape[0], dwi_image.shape[1], dwi_image.shape[2]))
        original_images[0] = dwi_image
        original_images[1] = nib.load(adc_path).get_fdata()
        original_images[2] = nib.load(flair_path).get_fdata()
        original_images[3] = nib.load(target_path).get_fdata()
        if self.transform is not None:
            #original_images[0:3] = histogram(original_images[0:3])
            original_images = self.transform(original_images)
            original_images[0:3] = normalize(original_images[0:3])

        mask = np.array([original_images[3] > 0.5]).astype(np.float32)

        input_data=torch.tensor(original_images[0:3])
        mask=torch.tensor(mask)

        patch_idx = index % self.num_patches
        start_indices = [
                patch_idx * (sz // self.num_patches) for sz in input_data[0].shape
        ]
        
        # Calculate the end indices for each dimension
        end_indices = [
            start_indices[dim] + self.patch_size[dim] for dim in range(len(start_indices))
        ]
        
        # Extract the patch
        channel_patch=[]
        for i in range(len(original_images)-1):
            channel_patch.append(input_data[i][
                start_indices[0]:end_indices[0],
                start_indices[1]:end_indices[1],
                start_indices[2]:end_indices[2],
            ])

        mask_patch=mask[0][
                start_indices[0]:end_indices[0],
                start_indices[1]:end_indices[1],
                start_indices[2]:end_indices[2],
            ]

        combined_patch = torch.stack(channel_patch)

        return combined_patch, mask_patch
    
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

