import os

import torchio as tio
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from loss import BCEDiceLoss

from Model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    crop_image,
)

# Hyperparameters etc
LEARNING_RATE = 1E-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 30
NUM_WORKERS = os.cpu_count()
IMAGE_HEIGHT = 80
IMAGE_WIDTH = 80
IMAGE_DEPTH = 80
CROP = [1,1,1]
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "Dataset001_ISLES22forUNET_Debug_L/imagesTr"
TRAIN_MASK_DIR = "Dataset001_ISLES22forUNET_Debug_L/labelsTr"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    """
    Train the model for one epoch
    Parameters
    ----------
    loader: A dataloader of the training set
    model: The model to train
    optimizer: The optimizer to use
    loss_fn: The loss function to use
    scaler: The scaler to use for mixed precision training
    -------

    """
    loop = tqdm(loader)
    batch_idx = 0
    avg_loss = 0.0
    for data, targets in loop:
        total_loss = 0.0
        number_iter = 0
        crop_data, crop_targets = crop_image(data, targets)
        for i in range(CROP[0]):
            for j in range(CROP[1]):
                for k in range(CROP[2]):
                    data = crop_data[:,:, i,j,k,:,:,:]
                    targets = crop_targets[:, :, i,j,k,:,:,:]
                    data = data.to(device=DEVICE)
                    targets = targets.float().to(device=DEVICE)

                    # forward
                    with torch.cuda.amp.autocast():
                        predictions = model(data)
                        loss = loss_fn(predictions, targets)

                    # backward
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.step()

                    # update tqdm loop
                    number_iter += 1
                    total_loss += loss.item()
                    avg_loss = total_loss/number_iter
        loop.set_postfix(loss=avg_loss)
    batch_idx += 1

def create_model(model):
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()

def main():

    # define transforms to augment the data
    patch_size = (IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH)

    #transform of a 3D image.
    train_transform = tio.Compose([
        #Random rotation of 10 degrees
        tio.RandomAffine(scales=1, degrees=[-10, 10, -10, 10, -10, 10], isotropic=True, image_interpolation='nearest'),
        tio.ToCanonical(),
        tio.Resample(4),
        tio.CropOrPad((IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH)),
        tio.RandomMotion(p=0.2),
        tio.RandomBiasField(p=0.3),
        tio.RandomNoise(p=0.5),
        tio.RandomFlip(),
        #Normalization occurs later
    ])
    val_transform = tio.Compose([
        tio.ToCanonical(),
        tio.CropOrPad((IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH)),
    ])

    #model definition
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = BCEDiceLoss(0.3, 0.7, 50.0, DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=LEARNING_RATE/NUM_EPOCHS)

    create_model(model)

    #creating the path for images. Train has 3 channels, Mask has 1
    train_files = []
    mask_files = []
    for i, filename in enumerate(sorted(os.listdir(TRAIN_IMG_DIR))):
        if i % 3 == 0:
            train_files.append([os.path.join(TRAIN_IMG_DIR, filename)])
        else:
            train_files[-1].append(os.path.join(TRAIN_IMG_DIR, filename))
    for filename in sorted(os.listdir(TRAIN_MASK_DIR)):
        mask_files.append(os.path.join(TRAIN_MASK_DIR, filename))

    #splitting test, train and validation
    df = pd.DataFrame(data={"filename": train_files, 'mask': mask_files})
    df_train, df_test = train_test_split(df, test_size=0.2)
    df_train, df_val = train_test_split(df_train, test_size=0.2)

    #Creating Dataloaders
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

    #Traing in batches, save every 10 epochs
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        # print some examples to a folder
        if(epoch%5 == 0):
            save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
            )
            check_accuracy(val_loader, model, device=DEVICE)
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, "checkpoints")
            
    save_predictions_as_imgs(
        val_loader, model, folder="saved_images/", device=DEVICE
    )
    check_accuracy(val_loader, model, device=DEVICE)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, "final_checkpoint")

if __name__ == "__main__":
    main()