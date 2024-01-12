import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import numpy as np
import cv2

# Load the trained model
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
model.eval()

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    return image

# Function to perform segmentation on the uploaded image
def perform_segmentation(image):
    with torch.no_grad():
        image = preprocess_image(image)
        output = model(image).sigmoid().squeeze().cpu().numpy()
    return output

# Streamlit app
def main():
    st.title("Cell Segmentation App")

    st.write("## Dataset Information:")
    st.write("The dataset contains 180 high-resolution color microscopic images of human duodenum adenocarcinoma HuTu 80 cell populations.")
    st.write("Images were captured using an in vitro scratch assay experimental protocol.")
    st.write("Images are taken at different time points: immediately following scratch formation, and after 24, 48, and 72 hours of cultivation.")
    st.write("Images were obtained with the Zeiss Axio Observer 1.0 microscope with 400x magnification.")
    st.write("All images have been manually annotated by domain experts, and these manual annotations serve as the ground truth for segmentation.")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)

        # Perform segmentation
        segmented_mask = perform_segmentation(image)

        # Display original and segmented images side by side
        col1, col2 = st.columns(2)
        col1.image(image, caption="Uploaded Image", use_column_width=True)
        col2.image(segmented_mask, caption="Segmentation Mask", use_column_width=True, channels="GRAY")

if __name__ == "__main__":
    main()
