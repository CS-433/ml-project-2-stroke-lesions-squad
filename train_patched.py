import os

import numpy as np
import torchio as tio
import torch
from monai.losses import DiceCELoss
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from monai import losses
from loss import DiceBCELoss_2

from Model import UNET
from utils import (
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)
from dataset import MRIImage, get_train_val_test_Dataloaders
from config import (
    DEVICE, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH, LEARNING_RATE, NUM_EPOCHS, PATCH_SIZE, CHECKPOINT_DIR, SAVED_IMAGES_DIR
)


def train_metrics(predictions, targets):
    """
    Calculate the accuracy and f1 score of the model
    Parameters
    ----------
    predictions: The predictions of the model of shape (BATCH_SIZE, 1, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH)
    targets: The ground truth of shape (BATCH_SIZE, 1, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH)
    Returns: The accuracy and f1 score
    -------
    """
    predictions = (predictions > 0.5).long()
    targets = (targets > 0.5).long()
    tp = torch.logical_and(predictions == 1, targets == 1).sum().item()
    tn = torch.logical_and(predictions == 0, targets == 0).sum().item()
    fp = torch.logical_and(predictions == 1, targets == 0).sum().item()
    fn = torch.logical_and(predictions == 0, targets == 1).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn+ 1e-10)
    return accuracy, f1, tp, tn, fp, fn


def train_fn_patched(loader, model, optimizer, loss_fn, scaler):
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
    model.train()
    loop = tqdm(loader)
    avg_loss = 0.0
    batch_accuracy, batch_f1, batch_tp, batch_tn, batch_fp, batch_fn = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    number_iter = 0
    total_loss = 0.0
    for data, targets in loop:
        
        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data.float())
            loss = loss_fn(predictions, targets.unsqueeze(1)).to(device=DEVICE)

        if np.isnan(loss.item()):
            print("Nan loss encountered")
            print(model(data))
            exit(1)
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        number_iter += 1
        total_loss += loss.item()
        loop.set_postfix(loss=total_loss / (number_iter + 1))
        accuracy, f1, tp, tn, fp, fn = train_metrics(predictions, targets)
        batch_accuracy += accuracy
        batch_f1 += f1
        batch_tp += tp
        batch_tn += tn
        batch_fp += fp
        batch_fn += fn

    return total_loss/number_iter, batch_accuracy/number_iter, batch_f1/number_iter, batch_tp, batch_tn, batch_fp, batch_fn


def main(backup_rate = 5):
    #transform of a 3D image.

    train_transform = tio.Compose([
        tio.RandomAffine(p=0.3),
        tio.RandomAnisotropy(p=0.1),
        tio.Blur(std=0.5, p=0.25),
        #tio.RandomMotion(degrees=15, translation=5, p=0.3),
        #tio.RandomBiasField(p=0.2),
        tio.RandomFlip(p=0.3),
        #tio.RandomElasticDeformation(max_displacement=10, p=0.05),
        tio.RandomSwap(p=0.3),
        # Normalization occurs later
    ])
    val_transform = tio.Compose([
        tio.RandomFlip(p=0.3),
    ])

    #model definition
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    loss_fn = DiceCELoss(sigmoid=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=LEARNING_RATE/NUM_EPOCHS)

    #Creating Dataloaders
    train_loader, val_loader, _ = get_train_val_test_Dataloaders(train_transform, val_transform, val_transform)

    scaler = torch.cuda.amp.GradScaler()

    losses = np.zeros(NUM_EPOCHS)
    metrics = {"train" : {"f1": [], "accuracy": [], "tp": [], "tn": [], "fp": [], "fn": []},
               "val": {"f1": [], "accuracy": [], "tp": [], "tn": [], "fp": [], "fn": []}}
    #Traing in batches, save every 10 epochs
    for epoch in range(NUM_EPOCHS):
        losses[epoch], accuracy, f1, tp, tn, fp, fn = train_fn_patched(train_loader, model, optimizer, loss_fn, scaler)
        metrics["train"]["f1"].append(f1)
        metrics["train"]["accuracy"].append(accuracy)
        metrics["train"]["tp"].append(tp)
        metrics["train"]["tn"].append(tn)
        metrics["train"]["fp"].append(fp)
        metrics["train"]["fn"].append(fn)
        # print some examples to a folder
        if(epoch%backup_rate == 0):
            save_predictions_as_imgs(
                val_loader, model, PATCH_SIZE, epoch, folder=SAVED_IMAGES_DIR, device=DEVICE)

            accuracy, f1, tp, tn, fp, fn = check_accuracy(val_loader, model, PATCH_SIZE, device=DEVICE)
            metrics["val"]["f1"].append(f1)
            metrics["val"]["accuracy"].append(accuracy)
            metrics["val"]["tp"].append(tp)
            metrics["val"]["tn"].append(tn)
            metrics["val"]["fp"].append(fp)
            metrics["val"]["fn"].append(fn)

            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, CHECKPOINT_DIR, epoch)
            
    save_predictions_as_imgs(
        val_loader, model, PATCH_SIZE,"final", folder=SAVED_IMAGES_DIR, device=DEVICE
    )
    accuracy, f1, tp, tn, fp, fn = check_accuracy(val_loader, model, device=DEVICE)
    metrics["val"]["f1"].append(f1)
    metrics["val"]["accuracy"].append(accuracy)
    metrics["val"]["tp"].append(tp)
    metrics["val"]["tn"].append(tn)
    metrics["val"]["fp"].append(fp)
    metrics["val"]["fn"].append(fn)

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, CHECKPOINT_DIR, "final")
    return losses, metrics

if __name__ == "__main__":
    main()