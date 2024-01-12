# Cell Populations Segmentation

This repository contains code for a deep learning model to segment cell populations in biomedical images. The code is organized into several modules to handle data, image processing, visualization, data preprocessing, model training, and evaluation.

## Modules

### 1. Data Handling (data_handling.py)
Handles loading and organizing the dataset. It includes functions to load image and mask paths, and split the dataset into training, validation, and test sets.

### 2. Image Processing (image_processing.py)
Contains functions for image processing, such as loading and saving images using OpenCV, and implementing edge density estimation.

### 3. Visualization (visualization.py)
Provides functions for visualizing images, masks, and the results of the segmentation process.

### 4. Data Preprocessing (data_preprocessing.py)
Handles the preprocessing of images and masks for training the model. Includes transformations for resizing and normalization.

### 5. Model Training (model_training.py)
Defines the model training process, including functions for training and validation steps, early stopping, and the overall training loop.

### 6. Evaluation (evaluation.py)
Contains functions for making predictions on the test set, calculating Intersection over Union (IoU) scores, and visualizing the results.

### 7. Main Script (main.py)
A simple script that utilizes the functions from the above modules to load data, train a segmentation model, evaluate its performance, and visualize the results.


### Results
![ori](https://github.com/YugantGotmare/human-duodenum-adenocarcinoma-HuTu-80-cell-Segmentation/assets/101650315/2c9ef907-f053-46d8-a16f-136576cadc93) ![pred](https://github.com/YugantGotmare/human-duodenum-adenocarcinoma-HuTu-80-cell-Segmentation/assets/101650315/c7177836-3bab-4792-ae2d-d3ce3e092ee4) 
