import gzip
import os
import shutil

import torch
from tqdm import tqdm

LEARNING_RATE = 1E-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8

NUM_EPOCHS = 50
NUM_WORKERS = os.cpu_count()
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_DEPTH = 128
PATCH_SIZE = (64,64,64)
NUM_PATCHES = 4
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "Dataset001_ISLES22forUNET_Debug/imagesTr"
TRAIN_MASK_DIR = "Dataset001_ISLES22forUNET_Debug/labelsTr"
CHECKPOINT_DIR = "checkpoint"
SAVED_IMAGES_DIR = "saved_images"


def uncompress():
    o_path = "Dataset001_ISLES22forUNET/labelsTr"
    n_path = "Dataset001_ISLES22forUNET_uc/labelsTr"
    path = os.listdir(o_path)
    for i in tqdm(range(205, len(path))):
        with gzip.open(os.path.join(o_path, path[i]), 'rb') as f_in:
            with open(os.path.join(n_path, path[i][:-3]), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
if __name__ == "__main__":
    uncompress()
