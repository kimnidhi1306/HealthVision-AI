# visualization.py

import matplotlib.pyplot as plt
from PIL import Image
import cv2
import segmentation_models_pytorch as smp

def im_show_cv2(image_np, title=None, cmap=None):
    plt.figure(figsize=(12, 15))
    plt.axis('off')
    plt.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), cmap=cmap)
    if title:
        plt.title(title)
    plt.show()

def im_show_gray(image_np, title=None, cmap="gray"):
    plt.figure(figsize=(12, 15))
    plt.axis('off')
    plt.imshow(image_np, cmap=cmap)
    if title:
        plt.title(title)
    plt.show()

def edge_density_plot(edges_np):
    plt.figure(figsize=(12, 15))
    plt.axis('off')
    plt.imshow(edges_np)
    plt.show()

def show_images_and_masks(image_paths, mask_paths, num_images=10):
    fig, ax = plt.subplots(nrows=num_images, ncols=2, figsize=(10, 30))
    
    for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        if i >= num_images:
            break
        
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax[i, 0].imshow(img_rgb)
        ax[i, 0].set_title(f'Image\nShape: {img_rgb.shape}')
        ax[i, 0].axis('off')
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        ax[i, 1].imshow(mask)
        ax[i, 1].set_title(f'Mask\nShape: {mask.shape}')
        ax[i, 1].axis('off')

    fig.tight_layout()
    fig.show()

def show_images_with_overlay(image_paths, mask_paths, num_images=20):
    fig, ax = plt.subplots(nrows=num_images, ncols=2, figsize=(10, 30))
    ax = ax.flat

    for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        if i >= num_images:
            break
        
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax[i].imshow(img_rgb)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        ax[i].imshow(mask, alpha=0.3)
        ax[i].axis('off')

    fig.tight_layout()
    fig.show()

def dataframe_head(data):
    data.head()
