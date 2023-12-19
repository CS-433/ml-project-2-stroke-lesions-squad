import nibabel
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import os
import nibabel as nib
CROP = [2,2,2]


def save_checkpoint(state,checkpoint_dir, epoch):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.

    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pytorch')
    torch.save(state, last_file_path)

def load_checkpoint(checkpoint_path, model, optimizer=None,
                    model_key='state_dict', optimizer_key='optimizer'):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state[model_key])

    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])

    return state

def crop_image(image, mask):
    """
    Crop the image and the mask to the size of the model
    Parameters
    ----------
    image : The image to crop of shape (BATCH_SIZE, 3, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH)
    mask: The mask to crop of shape (BATCH_SIZE, 1, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH)

    Returns : The cropped image and mask of shape (BATCH_SIZE, N_CHANNELS, CROP_DEPTH_DIVISOR, CROP_HEIGHT_DIVISOR, CROP_WIDTH_DIVISOR, CROP_DEPTH, CROP_HEIGHT, CROP_WIDTH)
    -------

    """
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

def f1(tp, fp, fn):
    return 2*tp/(2*tp+fp+fn)

def check_accuracy(loader, model, crop_patch_size, device="cuda"):
    num_correct = 0
    num_pixels = 0
    model.eval()

    tp, tn, fp, fn = 0, 0, 0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        sx, sy, sz = crop_patch_size[0], crop_patch_size[1], crop_patch_size[2]
        for i in range(0, y.shape[1], sx):
            for j in range(0, y.shape[2], sy):
                for k in range(0, y.shape[3], sz):
                    crop_x = x[:, :, i:i + sx, j:j + sy, k:k + sz]
                    crop_y = y[:, i:i + sx, j:j + sy, k:k + sz]
                    binary_y = (crop_y > 0.5).float()
                    with torch.no_grad():
                        preds = torch.sigmoid(model(crop_x.float()))
                        preds = (preds > 0.5).float()

                    f_tp, f_tn, f_fp, f_fn = bayesian(preds, binary_y)
                    tp += f_tp
                    tn += f_tn
                    fp += f_fp
                    fn += f_fn

                    num_correct += tp + tn
                    num_pixels += tp+ tn + fp + fn


    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"True Positive: {tp}, True Negative: {tn}, False Positive: {fp}, False Negative: {fn}")
    print(f"f1 score: {f1(tp, fp, fn)}")
    model.train()
    return num_correct / num_pixels, f1(tp, fp, fn), tp, tn, fp, fn

def save_predictions_as_imgs(
    loader, model, crop_patch_size, epoch, folder="saved_images/", device="cuda"
):
    """
    Save the predictions of the model on the loader in the folder. Saves one image per batch.
    Parameters
    ----------
    loader : The loader to use, one iteration of the loader must return the image and the mask of shape (BATCH_SIZE, 3, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH) and (BATCH_SIZE, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH) respectively
    model : The model to use
    crop_patch_size : The size of the patch to crop
    epoch : The epoch to save the images
    folder : The folder to save the images, defaults to "saved_images/"
    device : The device to use, defaults to "cuda"
    """
    model.eval()
    batch_idx = 0

    subfolder = f"{folder}/epoch_{epoch}"
    if not os.path.exists(subfolder):
        os.mkdir(subfolder)

    for x, y in loader:
        x = x.to(device=device)
        y = y.to(device=device)
        true_image = y[0]
        full_pred = torch.zeros(y.shape[1], y.shape[2], y.shape[3])
        sx, sy, sz = crop_patch_size[0], crop_patch_size[1], crop_patch_size[2]
        for i in range(0, y.shape[1], sx):
            for j in range(0, y.shape[2], sy):
                for k in range(0, y.shape[3], sz):
                    x_crop = x[:,:, i:i+sx,j:j+sy,k:k+sz]
                    with torch.no_grad():
                        preds = torch.sigmoid(model(x_crop.float()))
                        preds = (preds > 0.5).float()
                    full_pred[i:i+sx,j:j+sy,k:k+sz] = preds[0, 0]

        for slice in range(0, full_pred.shape[0], full_pred.shape[0] // 4):
            pred_image = full_pred[slice]
            y_image = true_image[slice]
            torchvision.utils.save_image(pred_image, f"{subfolder}/pred_{batch_idx}_slice{slice}.png")
            torchvision.utils.save_image(y_image, f"{subfolder}/y_{batch_idx}_slice{slice}.png")
            nib.save(nib.nifti1.Nifti1Image(pred_image.cpu().numpy(), np.eye(4)), f"{subfolder}/pred_{batch_idx}_slice{slice}.nii.gz")
            nib.save(nib.nifti1.Nifti1Image(y_image.cpu().numpy(), np.eye(4)), f"{subfolder}/y_{batch_idx}_slice{slice}.nii.gz")
        batch_idx += 1
    model.train()