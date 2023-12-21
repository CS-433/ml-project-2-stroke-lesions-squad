###FILE ORGANIZATION

config.py : here you can modify all the important hyperparameters and the different data paths
utils.py : file for for practical functions : computing the metrics, saving images, resizing and organizing the data loading
the function load_checkpoint can be used to load a pretrained model
Loss.py :  loss function used in the training, the default set for the combined loss is 0.25 dice score + 0.75 BCE
Dataset.py : Dataloader class (Explain how to organize the files)
Main.py : training function, saving checkpoints, setting all the transformations for data augementation
Run.py : running the training and saving the metrics





[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/fEFF99tU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=12937603&assignment_repo_type=AssignmentRepo)
