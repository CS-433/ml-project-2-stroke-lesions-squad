#Imports
import os
import random
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            #ConvTranspose3d is more expensive in terms of memory and time
            self.ups.append(nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2,))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, img):
        """
        Forward pass of the UNet
        Parameters
        ----------
        img : The input image of shape (BATCH_SIZE, 3, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH)

        Returns: The output of the UNet of shape (BATCH_SIZE, 1, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH)
        -------

        """
        # Connection is the list of outputs from the downsampling path.
        # We save it to keep local information. So where is the information
        connections = []
        #path down the UNet, finds important informations
        for down in self.downs:
            img = down(img)
            connections.append(img)
            img = self.pool(img)

        #link from downsampling to upsampling
        img = self.bottleneck(img)

        #reverse the connections list to go up the UNet
        connections = connections[::-1]
        for idx in range(0, len(self.ups), 2):
            #ConvTranspose3d is the upsampling layer
            img = self.ups[idx](img)
            #concatenates the output from the upsampling layer with the output from the downsampling layer
            connection = connections[idx//2]
            if(img.shape != connection.shape):
                connection = TF.resize(connection, size=img.shape[2:])
            #concatenat the image slices with the connection, along the channel axis
            img = torch.cat((img, connection), dim=1)
            #DoubleConv is the downsampling layer
            img = self.ups[idx+1](img)

        x = self.final_conv(img)

        return x