import copy
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from sklearn.utils import shuffle
from config import (
    TRAIN_IMG_DIR, TRAIN_MASK_DIR, PATCH_SIZE, NUM_PATCHES,
    TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE,
    NUM_WORKERS, PIN_MEMORY
)
import torchvision.transforms.functional as TF

class MRIImage(Dataset):
    def __init__(self, image_paths, labels_paths, transform=None, split_ratios=[0.8, 0.2, 0.1], mode = None , patch_size = (64, 64, 64),num_patches = 2):
        """
        Create a dataset from a dataframe of images and labels.
        Parameters
        ----------
        image_paths : The paths to the images in the dataset of shape (NUM_IMAGES * NUM_CHANNELS)
        labels_paths : The paths to the labels in the dataset of shape (NUM_IMAGES)
        """
        super(MRIImage, self).__init__()
        self.transform = transform
        self.split_ratios = split_ratios
        self.mode = mode

        images_list = np.array([os.path.join(image_paths, x) for x in os.listdir(image_paths)])
        labels_list = np.array([os.path.join(labels_paths, x) for x in os.listdir(labels_paths)])


        num_channels = len(images_list) // len(labels_list)
        num_training_imgs = len(images_list) // num_channels

        train_val_test = [int(x * num_training_imgs) for x in split_ratios]

        self.labels_paths = np.sort(labels_list)
        self.image_paths = np.sort(images_list).reshape(-1, num_channels)
        selected = np.arange(0, num_training_imgs)
        selected = shuffle(selected)

        self.train_image_path = self.image_paths[selected[:train_val_test[0]]] #might want to add .values
        self.train_label_path = self.labels_paths[selected[:train_val_test[0]]]
        self.val_image_path = self.image_paths[selected[train_val_test[0]:train_val_test[0] + train_val_test[1]]]
        self.val_label_path = self.labels_paths[selected[train_val_test[0]:train_val_test[0] + train_val_test[1]]]
        self.test_image_path = self.image_paths[selected[train_val_test[0] + train_val_test[1]:]]
        self.test_label_path = self.labels_paths[selected[train_val_test[0] + train_val_test[1]:]]

        self.patch_size = patch_size  # Each channel is 128x128x128, patches will be 64x64x64
        self.num_patches = num_patches

    def set_mode(self, mode):
        if mode != "train" and mode != "val" and mode != "test":
            raise ValueError("mode must be either train, val or test")
        self.mode = mode

    def __len__(self):
        if self.mode == "train":
            return len(self.train_label_path) * self.num_patches
        elif self.mode == "val":
            return len(self.val_label_path)
        elif self.mode == "test":
            return len(self.test_label_path) * self.num_patches

    def __getitem__(self, index):
        """
        Get an image and its label from the dataset using an index.
        Parameters
        ----------
        index : The index of the image to get

        Returns
        -------
        original_images: The original images of shape (3, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH)
        mask: The mask of shape (IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH)
        -------

        """

        #select the correct mode
        if self.mode == "train":
            image_paths = self.train_image_path[index//self.num_patches]
            label_path = self.train_label_path[index//self.num_patches]
            transform = self.transform[0]
        elif self.mode == "val":
            image_paths = self.val_image_path[index]
            label_path = self.val_label_path[index]
            transform = self.transform[1]
        elif self.mode == "test":
            image_paths = self.test_image_path[index//self.num_patches]
            label_path = self.test_label_path[index//self.num_patches]
            transform = self.transform[2]
        else:
            raise ValueError("mode must be either train, val or test")

        dwi_path = image_paths[0]
        adc_path = image_paths[1]
        flair_path = image_paths[2]

        dwi_image = nib.load(dwi_path).get_fdata()

        original_images = np.zeros((4, dwi_image.shape[0], dwi_image.shape[1], dwi_image.shape[2]))
        original_images[0] = dwi_image
        original_images[1] = nib.load(adc_path).get_fdata()
        original_images[2] = nib.load(flair_path).get_fdata()
        original_images[3] = nib.load(label_path).get_fdata()

        if transform is not None:
            #original_images[0:3] = histogram(original_images[0:3])
            original_images = transform(original_images)
            original_images[0:3] = normalize(original_images[0:3])

        mask = np.array([original_images[3] > 0.5]).astype(np.float32)

        if self.mode == "val":
            return original_images[0:3], mask[0]

        input_data = torch.tensor(original_images[0:3])
        mask = torch.tensor(mask)

        patch_idx = index % self.num_patches
        start_indices = [
            patch_idx * (sz // self.num_patches) for sz in input_data[0].shape
        ]

        # Calculate the end indices for each dimension
        end_indices = [
            start_indices[dim] + self.patch_size[dim] for dim in range(len(start_indices))
        ]

        # Extract the patch
        channel_patch = []
        for i in range(len(original_images) - 1):
            channel_patch.append(input_data[i][
                                 start_indices[0]:end_indices[0],
                                 start_indices[1]:end_indices[1],
                                 start_indices[2]:end_indices[2],
                                 ])

        for i in range(len(channel_patch)):
            if(channel_patch[i].shape != self.patch_size):
                channel_patch[i] = TF.resize(channel_patch[i], size=self.patch_size)

        mask_patch = mask[0][
                     start_indices[0]:end_indices[0],
                     start_indices[1]:end_indices[1],
                     start_indices[2]:end_indices[2],
                     ]
        if(mask_patch.shape != self.patch_size):
            mask_patch = TF.resize(mask_patch, size=self.patch_size)

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

def get_train_val_test_Dataloaders(train_transforms, val_transforms, test_transforms):
    dataset = MRIImage(TRAIN_IMG_DIR, TRAIN_MASK_DIR, [train_transforms, val_transforms, test_transforms], patch_size=PATCH_SIZE, num_patches=NUM_PATCHES)

    train_set, val_set, test_set = copy.deepcopy(dataset), copy.deepcopy(dataset), copy.deepcopy(dataset)
    train_set.set_mode('train')
    val_set.set_mode('val')
    test_set.set_mode('test')

    train_dataloader = DataLoader(dataset=train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=True)
    val_dataloader = DataLoader(dataset=val_set, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=True)
    test_dataloader = DataLoader(dataset=test_set, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=True)

    return train_dataloader, val_dataloader, test_dataloader
