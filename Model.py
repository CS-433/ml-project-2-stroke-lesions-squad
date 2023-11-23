#Imports
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import cv2
from tqdm.notebook import tqdm
from glob import glob
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from PIL import Image



def conv_layer(input_channels, output_channels):     #This is a helper function to create the convolutional blocks
    conv = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True)
    )
    return conv

class UNet(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],):
        super(UNet, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Down part of the UNet
        for feature in features:
            self.downs.append(conv_layer(in_channels, feature))
            in_channels = feature

        #Up part of the UNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(conv_layer(feature*2, feature))

        self.bottleneck = conv_layer(features[-1], features[-1]*2)

        self.output = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.output_activation = nn.Sigmoid()
                
    def forward(self, img):     #The print statements can be used to visualize the input and output sizes for debugging
        # Connection is the list of outputs from the downsampling path.
        # We save it to keep local information. So where is the information
        connections = []
        #path down the UNet, finds important informations
        for down in self.downs:
            img = down(img)
            connections.append(img)

            img = self.max_pool(img)

        #link from downsampling to upsampling
        img = self.bottleneck(img)

        #reverse the connections list to go up the UNet
        connections = connections[::-1]
        for idx in range(0, len(self.ups), 2):
            #ConvTranspose2d is the upsampling layer
            img = self.ups[idx](img)
            #concatenates the output from the upsampling layer with the output from the downsampling layer
            connection = connections[idx//2]
            if(img.shape != connection.shape):
                connection = TF.resize(connection, size=img.shape[2:])
            img = torch.cat((img, connection), dim=1)
            #convolutional layer
            img = self.ups[idx+1](img)

        x = self.output(img)
        x = self.output_activation(x)
        
        return x