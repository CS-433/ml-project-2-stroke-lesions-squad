import os

import torchio as tio
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from Model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 5
NUM_EPOCHS = 30
NUM_WORKERS = 8
IMAGE_HEIGHT = 80
IMAGE_WIDTH = 80
IMAGE_DEPTH = 16
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "Dataset001_ISLES22forUNET_Debug/imagesTr"
TRAIN_MASK_DIR = "Dataset001_ISLES22forUNET_Debug/labelsTr"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    batch_idx = 0
    for data, targets in loop:
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
        loop.set_postfix(loss=loss.item())
        batch_idx += 1


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
        #Normalization occurs later
    ])
    val_transform = tio.Compose([
        tio.Resize(patch_size),
    ])

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
            train_transform,
            val_transform,
            NUM_WORKERS,
            PIN_MEMORY,
            )

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        """ # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)"""

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        """if(epoch%3 == 0):
            save_predictions_as_imgs(
                val_loader, model, folder="saved_images/", device=DEVICE
            )"""


if __name__ == "__main__":
    main()