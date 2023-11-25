import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

from utils import get_loaders

from tqdm import tqdm

import Model

LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 16
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "Dataset001_ISLES22forUNET/magesTr"
TRAIN_MASK_DIR = "Dataset001_ISLES22forUNET/labelsTr"


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
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.compose([
        A.resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),

        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    model = Model.UNet().to(DEVICE)
    optimizer = optim.Adam(model.param
eters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn_dice = dc_loss
    df = pd.DataFrame(data={"filename": TRAIN_IMG_DIR, 'mask': TRAIN_MASK_DIR})
    df_train, df_test = train_test_split(df, test_size=0.2)
    df_train, df_val = train_test_split(df_train, test_size=0.2)


    train_loader = get_loaders(
            df_train["filename"],
            df_train["mask"],
            df_val["filename"],
            df_val["mask"],
            BATCH_SIZE,
            NUM_WORKERS,
            PIN_MEMORY,
            train_transform)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        #save model
        #check accuracy
        #print examples to a folder

if __name__ == "__main__":
    main()