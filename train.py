import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
import torchio as tio
from sklearn.model_selection import train_test_split
from utils import get_loaders, check_accuracy
import cProfile
import pstats

from tqdm import tqdm

import Model

LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 80
IMAGE_WIDTH = 80
IMAGE_DEPTH = 16
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "Dataset001_ISLES22forUNET_Debug/imagesTr"
TRAIN_MASK_DIR = "Dataset001_ISLES22forUNET_Debug/labelsTr"


def dc_loss(pred, target):
    smooth = 100

    predf = pred.view(-1)
    targetf = target.view(-1)
    intersection = (predf * targetf).sum()

    return 1 - ((2. * intersection + smooth) /
                (predf.sum() + targetf.sum() + smooth))

def train_fn(loader, model, optimizer, loss_fn, scaler):
    avg_train_losses = []
    avg_val_losses = []

    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.backward().item())


def main():
    # define transforms to augment the data
    patch_size = (IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH)
    train_transform = tio.Compose([
        #Random rotation of 10 degrees
        tio.RandomAffine(scales=1, degrees=[-10, 10, -10, 10, -10, 10], isotropic=True, image_interpolation='nearest'),
        tio.Resize(patch_size),
        tio.RandomFlip(0, p=0.5),
        tio.RandomFlip(1, p=0.5),
        tio.RandomFlip(2, p=0.5),
        tio.ZNormalization()
    ])
    val_transform = tio.Compose([
        tio.Resize(patch_size),
        tio.ZNormalization()
    ])

    # define model, optimizer, loss function
    model = Model.UNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    loss_fn_dice = dc_loss

    train_files = []
    mask_files = []
    for i, filename in enumerate(sorted(os.listdir(TRAIN_IMG_DIR))):
        if i % 3 == 0:
            train_files.append([os.path.join(TRAIN_IMG_DIR, filename)])
        else:
            train_files[-1].append(os.path.join(TRAIN_IMG_DIR, filename))
    for filename in sorted(os.listdir(TRAIN_MASK_DIR)):
        mask_files.append(os.path.join(TRAIN_MASK_DIR, filename))

    df = pd.DataFrame(data={"filename": train_files, 'mask': mask_files})
    df_train, df_test = train_test_split(df, test_size=0.2)
    df_train, df_val = train_test_split(df_train, test_size=0.2)


    train_loader, val_loader = get_loaders(
            df_train["filename"],
            df_train["mask"],
            df_val["filename"],
            df_val["mask"],
            BATCH_SIZE,
            NUM_WORKERS,
            PIN_MEMORY,
            train_transform,
            val_transform
            )

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        #save model
        #check accuracy
        acc = check_accuracy(val_loader, model, device=DEVICE)
        #print examples to a folder

if __name__ == "__main__":
    main()