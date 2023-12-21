#losses
import torch
import torch.nn as nn
from config import (
    TRAIN_IMG_DIR, TRAIN_MASK_DIR, PATCH_SIZE, NUM_PATCHES,
    TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE,
    NUM_WORKERS, PIN_MEMORY, DEVICE
)

def dice_coefficient(predicted, target, epsilon=1e-6):
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target)
    dice_score = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice_score

class DiceBCELoss_2(nn.Module):
    def __init__(self, device=DEVICE):
        super(DiceBCELoss_2, self).__init__()
        self.device = device

    def forward(self, predicted, target):
        # Ensure predicted and target tensors are of the same shape
        if predicted.shape != target.shape:
            predicted = predicted.squeeze(1)
        

        sig_predicted = nn.Sigmoid()(predicted)
        # Calculate Dice Loss
        dice_loss = 1 - dice_coefficient(sig_predicted, target)

        # Calculate Binary Cross Entropy Loss
        bce_loss = nn.BCEWithLogitsLoss()(predicted, target).to(self.device)

        # Combine both losses
        combined_loss = 0.25*dice_loss + 0.75*bce_loss

        return combined_loss