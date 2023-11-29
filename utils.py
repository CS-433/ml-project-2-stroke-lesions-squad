import torch
import torchvision
from dataset import MRIImage
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = MRIImage(
        train_dir,
        train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        persistent_workers=True
    )

    val_ds = MRIImage(
        val_dir,
        val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        persistent_workers=True
    )

    return train_loader, val_loader


def bayesian(preds, y):
    tp = torch.logical_and(preds == 1, y == 1).sum().item()
    tn = torch.logical_and(preds == 0, y == 0).sum().item()
    fp = torch.logical_and(preds == 1, y == 0).sum().item()
    fn = torch.logical_and(preds == 0, y == 1).sum().item()

    return tp, tn, fp, fn


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        tp, tn, fp, fn = 0, 0, 0, 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            binary_y = (y > 0.5)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == binary_y).sum()
            num_pixels += torch.numel(preds)

            f_tp, f_tn, f_fp, f_fn = bayesian(preds, binary_y)
            tp += f_tp
            tn += f_tn
            fp += f_fp
            fn += f_fn


    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"True Positive: {tp}, True Negative: {tn}, False Positive: {fp}, False Negative: {fn}")
    print(f"f1 score: {2*tp/(2*tp+fp+fn)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    idx = 0
    for x, y in loader:
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        for slice in range(0, x.shape[2], 8):
            torchvision.utils.save_image(
                preds[idx, 0, slice], f"{folder}/pred_{idx}_slice{slice}.png"
            )
            torchvision.utils.save_image(y[idx, slice], f"{folder}/y_{idx}_slice{slice}.png")
        idx += 1

    model.train()