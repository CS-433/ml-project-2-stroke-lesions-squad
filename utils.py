import torch
import torchvision
from dataset import MRIImage, MRIImage_patched
from torch.utils.data import DataLoader
import os
CROP = [2,2,2]


def save_checkpoint(state,checkpoint_dir):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.

    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    torch.save(state, last_file_path)

def load_checkpoint(checkpoint_path, model, optimizer=None,
                    model_key='model_state_dict', optimizer_key='optimizer_state_dict'):
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
    """

    Parameters
    ----------
    train_dir : The directory of the train file of shape (3, n_train)
    train_maskdir : The directory of the train mask file of shape (1, n_train)
    val_dir : The directory of the validation file of shape (3, n_val)
    val_maskdir : The directory of the validation mask file of shape (1, n_val)
    batch_size : The batch size
    train_transform : The transform for the train dataset
    val_transform : The transform for the validation dataset
    num_workers : The number of workers in parallel for the DataLoader, defaults to 4
    pin_memory : Whether to pin the memory, defaults to True

    Returns : train_loader a DataLoader for the train dataset, val_loader a DataLoader for the validation dataset
    -------

    """
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

def get_loaders_patched(
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
    """

    Parameters
    ----------
    train_dir : The directory of the train file of shape (3, n_train)
    train_maskdir : The directory of the train mask file of shape (1, n_train)
    val_dir : The directory of the validation file of shape (3, n_val)
    val_maskdir : The directory of the validation mask file of shape (1, n_val)
    batch_size : The batch size
    train_transform : The transform for the train dataset
    val_transform : The transform for the validation dataset
    num_workers : The number of workers in parallel for the DataLoader, defaults to 4
    pin_memory : Whether to pin the memory, defaults to True

    Returns : train_loader a DataLoader for the train dataset, val_loader a DataLoader for the validation dataset
    -------

    """
    train_ds = MRIImage_patched(
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

    val_ds = MRIImage_patched(
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

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
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
    print(f"f1 score: {f1(tp, fp, fn)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    """
    Save the predictions of the model on the loader in the folder. Saves one image per batch.
    Parameters
    ----------
    loader : The loader to use, one iteration of the loader must return the image and the mask of shape (BATCH_SIZE, 3, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH) and (BATCH_SIZE, 1, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH) respectively
    model : The model to use
    folder : The folder to save the images, defaults to "saved_images/"
    device : The device to use, defaults to "cuda"
    """
    model.eval()
    batch_idx = 0
    for x, y in loader:
        x = x.to(device=device)
        y = y.to(device=device)
        crop_img, crop_target = crop_image(x, y)
        for i in range(CROP[0]):
            for j in range(CROP[1]):
                for k in range(CROP[2]):
                    img = crop_img[:, :, i, j, k, :, :, :]
                    targ = crop_target[:, :, i, j, k, :, :, :]
                    with torch.no_grad():
                        preds = torch.sigmoid(model(img.float()))
                        preds = (preds > 0.5).float()

                    for slice in range(0, preds.shape[2] , preds.shape[2]//4):
                        pred_image = preds[0, 0, slice]
                        y_image = targ[0, 0, slice]
                        torchvision.utils.save_image(
                            pred_image, f"{folder}/pred_{batch_idx}_slice{slice}.png"
                        )
                        torchvision.utils.save_image(y_image, f"{folder}/y_{batch_idx}_slice{slice}.png")
        batch_idx += 1
    model.train()