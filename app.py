import os
import streamlit as st
import numpy as np
import io 
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow import keras
import segmentation_models_pytorch as smp
import torch
from torchvision import transforms
import pandas as pd
import pickle
from streamlit_option_menu import option_menu
from sklearn.ensemble import RandomForestClassifier

# Get the current working directory
base_path = os.path.dirname(os.path.abspath(__file__))

# Specify the relative path to the model.pkl file
diabetic_model_path = os.path.join(base_path, 'weights', 'model.pkl')
heart_disease_model_path = os.path.join(base_path, 'weights', 'heart.pkl')
kidney_model_path = os.path.join(base_path, 'weights', 'kidney.pkl')
breast_cancer_model_path = os.path.join(base_path, 'weights', 'Breast_cancer.pkl')
parkinson_model_path = os.path.join(base_path, 'weights', 'parkinsons.pkl')
segmentation_model_path = os.path.join(base_path, 'weights', 'segmentation_model_HuTu.pth')
classification_model_path = os.path.join(base_path, 'weights', 'kvasir_model.h5')
alzheimer_model_path = os.path.join(base_path, 'weights', 'Brain_Mri_Classifier_Alzheimer.h5')
covid_model_path = os.path.join(base_path, 'weights', 'covid.pkl')
diabetic_retinopathy_path = os.path.join(base_path, 'weights', 'diabetic-retinopathy.h5')

# Load the kvasir_main_model model for image classification
classification_model = load_model(classification_model_path, compile=False)
classification_model.layers[0].name = 'Conv_BatchNorm'  # Rename the layer
class_labels = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum',
                'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']

# Load the alzheimer_model for image classification
alzheimer_model = load_model(alzheimer_model_path, compile=False)
alzheimer_class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# Load the COVID-19 prediction model
with open(covid_model_path, 'rb') as file:
    knn_cv = pickle.load(file)

# Load the trained models for health prediction
diabetic_model = pickle.load(open(diabetic_model_path, 'rb'))
heart_disease_model = pickle.load(open(heart_disease_model_path, 'rb'))
kidney_model = pickle.load(open(kidney_model_path, 'rb'))
breast_cancer_model = pickle.load(open(breast_cancer_model_path, 'rb'))
parkinson_model = pickle.load(open(parkinson_model_path, 'rb'))

# Load the cell segmentation model
segmentation_model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
segmentation_model.load_state_dict(torch.load(segmentation_model_path, map_location=torch.device('cpu')))
segmentation_model.eval()

# Function to perform segmentation on the uploaded image
def perform_segmentation(image):
    with torch.no_grad():
        image = preprocess_image(image)
        output = segmentation_model(image).sigmoid().squeeze().cpu().numpy()
    return output

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    return image

# Function to make predictions on the image
def predict_image(image_path):
    image = preprocess_image(image_path)
    predictions = classification_model.predict(image)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

# Load and preprocess the image
def kvasir_preprocess_image(uploaded_file):
    # Convert the BytesIO object to a numpy array
    image_array = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # Decode the image
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # Resize and preprocess the image
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# Make predictions on the image
def kvasir_predict_image(image_path):
    image = kvasir_preprocess_image(image_path)
    predictions = classification_model.predict(image)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

# Function to predict the class
def diabetic_retinopathy_predict_class(image):
    image = np.array(image) / 255.0
    new_model = tf.keras.models.load_model(diabetic_retinopathy_path)  # Assuming it's a Keras model
    predictions = new_model.predict(image)
    predicted_class = np.argmax(predictions)  # Adjust this based on your model output
    return predicted_class

# Streamlit UI
def main():
    st.title("HealthVision AI")
    st.sidebar.title("Navigation")
    page_options = ["Home", "Diabetic Retinopathy Prediction", "Heart Disease Prediction", 
                    "Kidney Disease Prediction", "Breast Cancer Prediction", "Parkinson's Disease Prediction", 
                    "Cell Segmentation", "Alzheimer's Disease Prediction", "COVID-19 Prediction"]
    prediction_selected = option_menu("Navigate to", page_options)

    if prediction_selected == "Home":
        st.write("Welcome to HealthVision AI. Select a prediction task from the sidebar.")
    elif prediction_selected == "Diabetic Retinopathy Prediction":
        st.write("Diabetic Retinopathy Prediction")
        uploaded_file = st.file_uploader("Upload an image for prediction", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            if st.button('Predict'):
                prediction = diabetic_retinopathy_predict_class(image)
                st.write(f"Prediction: {prediction}")
    elif prediction_selected == "Heart Disease Prediction":
        st.write("Heart Disease Prediction")
        # Add your code for heart disease prediction here
    elif prediction_selected == "Breast Cancer Prediction":
        st.write("Breast Cancer Prediction")
        # Add your code for breast cancer prediction here
    elif prediction_selected == "Parkinson's Disease Prediction":
        st.write("Parkinson's Disease Prediction")
        # Add your code for Parkinson's disease prediction here
    elif prediction_selected == "Cell Segmentation":
        st.write("Cell Segmentation")
        # Add your code for cell segmentation here
    elif prediction_selected == "Alzheimer's Disease Prediction":
        st.write("Alzheimer's Disease Prediction")
        # Add your code for Alzheimer's disease prediction here
    elif prediction_selected == "COVID-19 Prediction":
        st.write("COVID-19 Prediction")
        # Add your code for COVID-19 prediction here

if __name__ == '__main__':
    main()
