#Imports
import os
import random
import time

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import gzip
import os
import shutil


from tqdm import tqdm

LEARNING_RATE = 5E-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 2
TEST_BATCH_SIZE = 2

NUM_EPOCHS = 300
BACKUP_RATE = 100
NUM_WORKERS = os.cpu_count()
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_DEPTH = 128
PATCH_SIZE = (64,64,64)
NUM_PATCHES = 4
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/kaggle/input/full-dataset/Dataset001_ISLES22forUNET_uc/imagesTr"
TRAIN_MASK_DIR = "/kaggle/input/full-dataset/Dataset001_ISLES22forUNET_uc/labelsTr"
CHECKPOINT_DIR = "/kaggle/working/checkpoint"
SAVED_IMAGES_DIR = "/kaggle/working/saved_images"