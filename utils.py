import torch
import torchvision
from dataset import MRIImage
from torch.utils.data import DataLoader
CROP = [2,2,2]


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
    split_img_dim,
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


def crop_image(image, mask):
    shape = image.size()
    cropped_shape = (shape[2] // CROP[0], shape[3] // CROP[1], shape[4] // CROP[2])

    crop = image.unfold(2, cropped_shape[0], cropped_shape[0]).unfold(3, cropped_shape[1],cropped_shape[1]).unfold(4,cropped_shape[2],cropped_shape[2])
    mask = mask.unfold(2, cropped_shape[0], cropped_shape[0]).unfold(3, cropped_shape[1], cropped_shape[1]).unfold(4,cropped_shape[2],cropped_shape[2])
    return crop, mask

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
            y = y.to(device)
            crop_img, crop_target = crop_image(x, y)
            for i in range(CROP[0]):
                for j in range(CROP[1]):
                    for k in range(CROP[2]):
                        x = crop_img[:,:, i,j,k,:,:,:]
                        y = crop_target[:, :, i,j,k,:,:,:]
                        binary_y = (y > 0.5).long()
                        with torch.no_grad():
                            preds = torch.sigmoid(model(x.float()))
                            preds = (preds > 0.5).long()
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
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device)
        crop_img, crop_target = crop_image(x, y)
        for i in range(CROP[0]):
            for j in range(CROP[1]):
                for k in range(CROP[2]):
                    x = crop_img[:, :, i, j, k, :, :, :]
                    y = crop_target[:, :, i, j, k, :, :, :]

                    with torch.no_grad():
                        preds = torch.sigmoid(model(x.float()))
                        preds = (preds > 0.5).float()

                    for slice in range(0, x.shape[2], 32):
                        pred_image = preds[idx, 0, slice]
                        y_image = y[idx, 0, slice]
                        torchvision.utils.save_image(
                            pred_image, f"{folder}/pred_{idx}_slice{slice}.png"
                        )
                        torchvision.utils.save_image(y_image, f"{folder}/y_{idx}_slice{slice}.png")

    model.train()