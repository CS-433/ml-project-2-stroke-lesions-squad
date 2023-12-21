# Stroke lesion segmentation using U-net architecture

In this project we aim to build a precise lesion segmentation tool designed for MRI images based on a 3D U-Net model.

## Description

Lesion segmentation in MRI plays a critical role in the early detection and characterization of various medical conditions, including neurodegenerative diseases, tumors, and vascular abnormalities. The process of segmentation is manual and thus time-consuming and subject to variability between annotators. 

The goal of this project is to provide a tool that can efficiently segmentate stroke lesions and provide a confidence level along with the prediction (a working uncertainty estimator has not been not implemented yet).

### Dataset

We used a publicly available dataset for lesion segmentation. The data is available on the ISLES22 challenge 

### Dependencies

The project is coded in Python 3 and Jupyter Notebook (full_train.ipynb is a ready-to-use notebook for training on Kaggle) using :
- mainly pytorch (torch, torchvision, torchio, torchmetrics)
- nibabel for Nifti MRI scans data handling

### File organization

- config.py : here you can modify all the important hyperparameters and the different data paths

- utils.py : file for practical functions (computing the metrics, saving images, resizing) and for organizing the data loading
the function load_checkpoint can be used to load a pretrained model

- Loss.py :  loss function used in the training, the default set for the combined loss is 0.25 dice score + 0.75 BCE

- Dataset.py : Dataloader class : made for Nifti data (3D data), we have 3 grayscale scans (FLAIR, DWI, ADC) per slice (thus the data treatment is similar to a 3D RGB image) for the 3 channel ibput of the model. The Nifti data files should be organized as the following : \
      patient0_dwi_scan.nii \
      patient0_adc_scan.nii  \
      patient0_flair_scan.nii  \
      patient1_dwi_scan.nii  \
      patient1_adc_scan.nii  \
      patient1_flair_scan.nii  \
      ... \
      

- Main.py : training function, saving checkpoints, setting all the transformations for data augmentation

- Run.py : running the training and saving the metrics

### Executing program

- Change the data paths (config.py and Run.py) and the hyperparameters (config.py)
- run the Run.py file for training
- Or load a pretrained model using the load_checkpoint function (utils.py) and directly use it for predictions





[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/fEFF99tU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=12937603&assignment_repo_type=AssignmentRepo)
