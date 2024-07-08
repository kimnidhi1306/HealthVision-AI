import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import segmentation_models_pytorch as smp
import torch
from torchvision import transforms
import pickle

# Get the current working directory
base_path = os.path.dirname(os.path.abspath(__file__))

# Specify the relative paths to model files
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

# Load the classification model for endoscopic images
classification_model = load_model(classification_model_path, compile=False)
classification_model.layers[0].name = 'Conv_BatchNorm'  # Rename the layer
class_labels = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum',
                'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']

# Load the Alzheimer's classification model
alzheimer_model = load_model(alzheimer_model_path, compile=False)
alzheimer_class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# Load the COVID-19 prediction model using pickle
with open(covid_model_path, 'rb') as file:
    knn_cv = pickle.load(file)

# Load the health prediction models using pickle
diabetic_model = pickle.load(open(diabetic_model_path, 'rb'))
heart_disease_model = pickle.load(open(heart_disease_model_path, 'rb'))
kidney_model = pickle.load(open(kidney_model_path, 'rb'))
breast_cancer_model = pickle.load(open(breast_cancer_model_path, 'rb'))
parkinson_model = pickle.load(open(parkinson_model_path, 'rb'))

# Load the cell segmentation model using segmentation_models_pytorch
segmentation_model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
segmentation_model.load_state_dict(torch.load(segmentation_model_path, map_location=torch.device('cpu')))
segmentation_model.eval()

# Function to perform segmentation on the uploaded image
def perform_segmentation(image):
    with torch.no_grad():
        image = preprocess_image(image)
        output = segmentation_model(image).sigmoid().squeeze().cpu().numpy()
    return output

# Function to preprocess an image for segmentation
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    return image

# Function to predict class for diabetic retinopathy from an image
def diabetic_retinopathy_predict_class(image):
    image = np.array(image) / 255.0  # Normalize the image
    new_model = tf.keras.models.load_model(diabetic_retinopathy_path)
    classes = new_model.predict_classes(image.reshape(1, 512, 512, 3))
    return classes

# Streamlit web app
def main():
    # Home page content
    home_page_content = """

    # Welcome to HealthVision AI

    Discover the future of personalized healthcare with HealthVision AI – your advanced health prediction and cell segmentation companion. Our web app leverages cutting-edge machine learning models to empower you with insights into various health conditions and intricate cell structures.

    ## Explore the Features:

    ### 🤖 **Health Prediction Systems:**
    Uncover predictions for diabetes, chronic kidney disease, breast cancer, heart disease, and Parkinson's disease. Input relevant data or upload medical images for accurate and personalized results.

    ### 🦠 **COVID-19 Prediction:**
    In these challenging times, HealthVision supports you with COVID-19 predictions. Input symptoms and relevant details to receive a prediction about the likelihood of COVID-19.

    ### 📸 **Endoscopic image Classification:**
    Dive into endoscopic image classification. Our model can identify features like dyed-lifted-polyps, esophagitis, and more, aiding in efficient categorization.

    ### 👁️ **Diabetic Retinopathy Detection:**
    Predict the presence of diabetic retinopathy in retinal images, offering a crucial tool for diabetes-related complications.

    ### 🧠 **Alzheimer's Brain Classification:**
    Upload brain images to our model, predicting the likelihood of Alzheimer's disease based on advanced machine learning techniques.

    ### 🎨 **Cell Segmentation:**
    Visualize the intricate world of cells with our segmentation model. Upload medical images, and witness the delineation of cell boundaries for in-depth analysis.

    ## How to Navigate:

    1. Explore different sections using the sidebar menu.
    2. Follow instructions on each page to input data or upload images.
    3. Uncover detailed predictions and segmentation results.

    ## About HealthVision AI:

    HealthVision brings together state-of-the-art machine learning models to provide you with powerful health insights. While our predictions are informed by data, they should not replace professional medical advice.

    Embark on a journey of discovery – navigate HealthVision, make predictions, and unlock the potential of personalized healthcare.

    Stay well, and enjoy your exploration with HealthVision!
    """

    # Sidebar menu for selecting tasks
    selected = st.sidebar.selectbox('Choose Task', ['About', 'Prediction Systems', 'COVID-19 Prediction', 
                                                   'Endoscopic image Classification',
                                                   'Diabetic Retinopathy Prediction', 
                                                   'Alzheimer Brain Classification', 'Cell Segmentation'])

    # Handle selection of tasks
    if selected == 'About':
        st.markdown(home_page_content)

    elif selected == 'Prediction Systems':
        # Sub-menu for selecting prediction systems
        prediction_selected = st.sidebar.selectbox('Prediction System',
                                                   ['Diabetes Prediction', 'Chronic Kidney Disease',
                                                    'Breast Cancer Prediction', 'Parkinson Disease Prediction'])

        # Handle different prediction systems
        if prediction_selected == 'Diabetes Prediction':
            st.subheader('Diabetes Prediction System')
            uploaded_image = st.file_uploader("Upload a retinal image for prediction", type=['png', 'jpg', 'jpeg'])
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                if st.button('Predict'):
                    with st.spinner('Predicting...'):
                        prediction = diabetic_retinopathy_predict_class(image)
                    st.success(f'The prediction is: {prediction}')

        elif prediction_selected == 'Chronic Kidney Disease':
            # Placeholder for Chronic Kidney Disease prediction system

        elif prediction_selected == 'Breast Cancer Prediction':
            # Placeholder for Breast Cancer Prediction system

        elif prediction_selected == 'Parkinson Disease Prediction':
            # Placeholder for Parkinson Disease Prediction system

    elif selected == 'COVID-19 Prediction':
        # Placeholder for COVID-19 Prediction system

    elif selected == 'Endoscopic image Classification':
        # Placeholder for Endoscopic image Classification system

    elif selected == 'Diabetic Retinopathy Prediction':
        st.subheader('Diabetic Retinopathy Prediction System')
        uploaded_image = st.file_uploader("Upload a retinal image for prediction", type=['png', 'jpg', 'jpeg'])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            if st.button('Predict'):
                with st.spinner('Predicting...'):
                    prediction = diabetic_retinopathy_predict_class(image)
                st.success(f'The prediction is: {prediction}')

    elif selected == 'Alzheimer Brain Classification':
        # Placeholder for Alzheimer Brain Classification system

    elif selected == 'Cell Segmentation':
        # Placeholder for Cell Segmentation system

# Run the Streamlit app
if __name__ == "__main__":
    main()
