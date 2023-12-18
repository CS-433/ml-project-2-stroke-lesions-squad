import gzip
import io
import os
import shutil

import nibabel
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchio as tio

from Model import UNET
from loss import DiceBCELoss_2
from utils import save_predictions_as_imgs, check_accuracy, load_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 5E-4
BATCH_SIZE = 4
NUM_EPOCHS = 30



class testDataset(Dataset):
    def __len__(self):
        return 4
    def __getitem__(self, index):
        x = torch.randn(3, 128, 128, 128)
        y = torch.randint(0, 1, (1, 128, 128, 128))
        return x,y

def test_save_image():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.nn.Sequential(
        torch.nn.Conv3d(3, 1, kernel_size=1, stride=1, padding=0, bias=False),
    ).to(DEVICE)
    loader = DataLoader(testDataset(), batch_size = 2)
    #check_accuracy(loader, model, (64,64,64), device=DEVICE)
    save_predictions_as_imgs(loader, model, (64,64,64), folder="saved_images/", device=DEVICE)

def test_load_model():
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    loss_fn = DiceBCELoss_2(device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=LEARNING_RATE/NUM_EPOCHS)

    load_checkpoint("final_checkpoint/last_checkpoint.pytorch", model, optimizer=optimizer)

    model.eval()

def test_extract():
    o_path = "Dataset001_ISLES22forUNET/imagesTr"
    n_path = "Dataset001_ISLES22forUNET_uc/imagesTr"
    path = os.listdir(o_path)
    for i in tqdm(range(len(path))):
        with gzip.open(os.path.join(o_path, path[i]), 'rb') as f_in:
            with open(os.path.join(n_path,path[i][:-3]), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    o_path = "Dataset001_ISLES22forUNET/labelsTr"
    n_path = "Dataset001_ISLES22forUNET_uc/labelsTr"
    path = os.listdir(o_path)
    for i in tqdm(range(len(path))):
        with gzip.open(os.path.join(o_path, path[i]), 'rb') as f_in:
            with open(os.path.join(n_path, path[i][:-3]), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

def test_reshape():
    img = torch.randn(3, 64, 64, 64)
    print(img.unsqueeze(1).shape)