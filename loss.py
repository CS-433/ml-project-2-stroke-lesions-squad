from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F


class Dice(nn.Module):
    def __init__(self, thresh: float = 0.5):
        """Dice coefficient."""
        super().__init__()
        assert 0 < thresh < 1, f"'thresh' must be in range (0, 1)"
        self.thresh = thresh

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                weights: Optional[torch.Tensor] = None, smooth: float = 0):
        # Binarize prediction
        inputs = torch.where(inputs < self.thresh, 0, 1)
        batch_size = targets.shape[0]

        intersection = torch.logical_and(inputs, targets)
        intersection = intersection.view(batch_size, -1).sum(-1)
        targets_area = targets.view(batch_size, -1).sum(-1)
        inputs_area = inputs.view(batch_size, -1).sum(-1)
        dice = (2. * intersection + smooth) / (inputs_area + targets_area + smooth)

        if weights is not None:
            assert weights.shape == dice.shape, \
                f'"weights" must be in shape of "{dice.shape}"'
            return (dice * weights).sum()

        return dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, thresh: float = 0.5, device = "cuda"):
        """Dice loss + binary cross-entropy loss."""
        super().__init__()
        assert 0 < thresh < 1, f"'thresh' must be in range (0, 1)"
        self.thresh = thresh
        self.dice = Dice(self.thresh)
        self.__name__ = 'DiceBCELoss'
        self.device = device

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                weights: Optional[torch.Tensor] = None, smooth: float = 0):
        batch_size = inputs.shape[0]

        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')

        if weights is not None:
            assert weights.shape == bce.shape, \
                f'"weights" must be in shape of "{bce.shape}"'
            bce = (bce * weights).sum()
        else:
            bce = bce.mean()

        dice_loss = 1 - self.dice(inputs, targets, weights, smooth)
        dice_bce = bce + dice_loss
        return dice_bce