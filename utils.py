#utils

#utils
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import os
#from torchmetrics.classification import *
from Loss import dice_coefficient
import torch.nn.functional as F


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

    state = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(state[model_key])

    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])

    return state


def train_metrics(predictions, targets, device):
    """
    Calculate the accuracy and f1 score of the model
    Parameters
    ----------
    predictions: The predictions of the model of shape (BATCH_SIZE, 1, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH)
    targets: The ground truth of shape (BATCH_SIZE, 1, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH)
    Returns: The accuracy and f1 score
    -------
    """
    if predictions.shape != targets.shape:
        predictions = predictions.squeeze(1)
    
    predictions = nn.Sigmoid()(predictions)
    predictions = (predictions > 0.5).long()
    targets = (targets > 0.5).long()
    tp = torch.logical_and(predictions == 1, targets == 1).sum().item()
    tn = torch.logical_and(predictions == 0, targets == 0).sum().item()
    fp = torch.logical_and(predictions == 1, targets == 0).sum().item()
    fn = torch.logical_and(predictions == 0, targets == 1).sum().item()
    dice=dice_coefficient(predictions,targets).item()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn+ 1e-10)
    return accuracy, f1, tp, tn, fp, fn, dice


def check_accuracy(loader, model, crop_patch_size, device="cuda"):
    """
    Check the accuracy and f1 score of the model on the loader
    Parameters
    ----------
    loader the validation loader
    model the model to use
    crop_patch_size the size of the patch to crop used before feeding the model
    device the device to use, defaults to "cuda"

    Returns : The accuracy and f1 score
    -------

    """
    num_correct = 0
    num_pixels = 0
    model.eval()

    tp, tn, fp, fn, f1, accuracy,dice = 0, 0, 0, 0, 0, 0,0
    num_iter = 0
    for x, y in loader:
        y = y.to(device)
        sx, sy, sz = crop_patch_size[0], crop_patch_size[1], crop_patch_size[2]
        #run over each patch
        for i in range(0, y.shape[1], sx):
            for j in range(0, y.shape[2], sy):
                for k in range(0, y.shape[3], sz):
                    crop_x = x[:, :, i:i + sx, j:j + sy, k:k + sz]
                    crop_y = y[:, i:i + sx, j:j + sy, k:k + sz]
                    binary_y = (crop_y > 0.5).float()
                    with torch.no_grad():
                        preds = compute_prediction(crop_patch_size, crop_x, model)

                    accuracy_t, f1_t, tp_t, tn_t, fp_t, fn_t,dice_t = train_metrics(preds, binary_y, device)
                    tp += tp_t
                    tn += tn_t
                    fp += fp_t
                    fn += fn_t
                    f1 += f1_t
                    accuracy += accuracy_t
                    dice+= dice_t

                    num_iter += 1


    print(
        f"Got average Accuracy : {accuracy/num_iter:.2f}"
    )
    print(f"True Positive: {tp}, True Negative: {tn}, False Positive: {fp}, False Negative: {fn}")
    print(f"Got average F1 score: {f1/num_iter:.4f}")
    model.train()

    return accuracy/num_iter, f1/num_iter, tp, tn, fp, fn, dice/num_iter


def compute_prediction(crop_patch_size, x, model):
    """
    Compute the prediction of the model on an image x
    Parameters
    ----------
    crop_patch_size the size of the patch to crop
    crop_x the crop to use of shape (BATCH_SIZE, 3, CROP_DEPTH, CROP_HEIGHT, CROP_WIDTH)
    model the model to use

    Returns
    -------

    """
    # resize the patch to the correct size. Computationaly expensive because it calls the cpu
    original_shape = x.shape
    crop_x = resize_tensor(x, crop_patch_size)
    # get the prediction
    preds = torch.sigmoid(model(crop_x.float()))
    preds = (preds > 0.5).float()
    preds = resize_tensor(preds, original_shape[2:])
    return preds

def resize_tensor(tensor: torch.Tensor, new_size):
    """
    Resize the tensor to the new size
    Parameters
    ----------
    tensor : The tensor to resize of shape (BATCH_SIZE, 3, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH)
    new_size : The new size of shape (IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH)

    Returns : The resized tensor
    -------

    """
    if tensor.shape[2:] == new_size:
        return tensor
    if tensor.shape[1:] == new_size:
        return tensor

    return F.interpolate(tensor, size=new_size, mode='trilinear', align_corners=False).to(tensor.device)

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
    if not os.path.exists(folder):
        os.mkdir(folder)

    subfolder = f"{folder}/epoch_{epoch}"
    if not os.path.exists(subfolder):
        os.mkdir(subfolder)

    for x, y in loader:
        x = x.to(device=device)
        y = y.to(device=device)
        true_image = y[0]
        full_pred = torch.zeros(y.shape[1], y.shape[2], y.shape[3])
        sx, sy, sz = crop_patch_size[0], crop_patch_size[1], crop_patch_size[2]
        #run over each patch
        for i in range(0, y.shape[1], sx):
            for j in range(0, y.shape[2], sy):
                for k in range(0, y.shape[3], sz):
                    crop_x = x[:, :, i:i + sx, j:j + sy, k:k + sz]
                    with torch.no_grad():
                        preds = compute_prediction(crop_patch_size, crop_x, model)

                    full_pred[i:i + sx, j:j + sy, k:k + sz] = preds[0, 0]

        for slice in range(0, full_pred.shape[0], full_pred.shape[0] // 4):
            save_image(batch_idx, full_pred, slice, subfolder)
            save_image(batch_idx, true_image, slice, subfolder)
        batch_idx += 1
        if batch_idx > 4:
            break
    model.train()


def save_image(batch_idx, img, slice, subfolder):
    pred_image = img[slice]
    torchvision.utils.save_image(pred_image, f"{subfolder}/pred_{batch_idx}_slice{slice}.png")
    
def log(metrics, index, epoch):
    """
    Log the metrics in a folder, creates the folder if it does not exist
    Parameters
    ----------
    metrics a dictionary containing the metrics, such as loss, f1, accuracy, tp, tn, fp, fn
    index  the index to log, either "train" or "val"
    -------
    """
    folder = f"logs/{index}"
    if not os.path.isdir(folder):
        os.makedirs(folder)
    metrics_list = []
    for key in metrics[index].keys():
        metrics_list.append(metrics[index][key])
    metrics_tensor = torch.tensor(metrics_list)
    torch.save(metrics_tensor, f"{folder}/metrics_epoch{epoch}.zip")