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
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
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
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            #ConvTranspose2d is more expensive in terms of memory and time
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2,))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, img):     #The print statements can be used to visualize the input and output sizes for debugging
        # Connection is the list of outputs from the downsampling path.
        # We save it to keep local information. So where is the information
        connections = []
        #reshapes the image into slices along x axis
        image_slices = img.reshape(img.shape[0]*img.shape[2], img.shape[1], img.shape[3], img.shape[4])
        #path down the UNet, finds important informations
        for down in self.downs:
            image_slices = down(image_slices)
            connections.append(image_slices)

            image_slices = self.pool(image_slices)

        #link from downsampling to upsampling
        image_slices = self.bottleneck(image_slices)

        #reverse the connections list to go up the UNet
        connections = connections[::-1]
        for idx in range(0, len(self.ups), 2):
            #ConvTranspose2d is the upsampling layer
            image_slices = self.ups[idx](image_slices)
            #concatenates the output from the upsampling layer with the output from the downsampling layer
            connection = connections[idx//2]
            if(image_slices.shape != connection.shape):
                connection = TF.resize(connection, size=image_slices.shape[2:])
            #concatenat the image slices with the connection, along the channel axis
            image_slices = torch.cat((image_slices, connection), dim=1)
            #convolutional layer
            image_slices = self.ups[idx+1](image_slices)

        x = self.final_conv(image_slices)
        reshaped = x.reshape(img.shape[0], 1, img.shape[2], img.shape[3], img.shape[4])

        return reshaped