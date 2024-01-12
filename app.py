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
class_labels = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum',
                'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']


# Load the alzheimer_model for image classification
alzheimer_model = load_model(alzheimer_model_path, compile=False)
alzheimer_class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']


# Load the COVID-19 prediction model
with open(covid_model_path, 'rb') as file:
    knn_cv = pickle.load(file)

# Load the trained models for health prediction
diabetic_model = pickle.load(open(diabetic_model_path , 'rb'))
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
    new_model = tf.keras.models.load_model(diabetic_retinopathy_path, compile = False)
    predict = new_model.predict(np.array([image]))
    per = np.argmax(predict, axis=1)
    return per[0]

# Function to predict COVID-19
def predict_covid(user_data):
    result = knn_cv.predict(user_data)
    return "Positive" if result[0] == 1 else "Negative"


# Streamlit app
def main():
    home_page_content = ""

    with st.sidebar:
        selected = option_menu('Choose Task', ['About', 'Prediction Systems', 'COVID-19 Prediction', 'Endoscopic image Classification', 'Diabetic Retinopathy Prediction', 'Alzheimer Brain Classification', 'Cell Segmentation'], default_index=0)

    if selected == 'About':
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

        # Display the home page content
    # st.markdown(home_page_content)


    if selected == 'Prediction Systems':
        prediction_selected = option_menu('Prediction System',
                                          ['Diabetes Prediction',
                                           'Chronic Kidney Disease',
                                           'Breast Cancer Prediction',
                                           'Parkinson Disease Prediction'],
                                          default_index=0)

        if prediction_selected == 'Diabetes Prediction':
             # page title
            st.write("### Diabetes Prediction System")
            st.write("Predict the likelihood of diabetes based on input features. Please provide the following information:")
            
            # getting the input data from the user
            col1, col2, col3 = st.columns(3)
            
            with col1:
                Pregnancies = st.text_input('Number of Pregnancies')
                
            with col2:
                Glucose = st.text_input('Glucose Level')
            
            with col3:
                BloodPressure = st.text_input('Blood Pressure value')
            
            with col1:
                SkinThickness = st.text_input('Skin Thickness value')
            
            with col2:
                Insulin = st.text_input('Insulin Level')
            
            with col3:
                BMI = st.text_input('BMI value')
            
            with col1:
                DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
            
            
            with col2:
                Age = st.text_input('Age of the Person')


            eature_types=['float', 'float', 'float', 'float', 'float', 'float', 'float', 'float']

            
            # code for Prediction
            diab_diagnosis = ''
            
            # creating a button for Prediction
            if st.button('Diabetes Test Result'):
                prediction = diabetic_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
                
                if (prediction[0] == 0):
                    diab_diagnosis = 'The person is not diabetic'
                else:
                    diab_diagnosis = 'The person is diabetic'
        
            st.success(diab_diagnosis)


        elif prediction_selected == 'Chronic Kidney Disease':
            # page title
            st.write("### Chronic Kidney Disease Prediction System")
            st.write("Predict the likelihood of chronic kidney disease based on input features. Please provide the following information:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.text_input('age')
                
            with col2:
                bp = st.text_input('blood_pressure')

            with col3:
                bgr = st.text_input('blood_glucose_random')
                
            
            with col3:
                al = st.text_input('albumin')
                
                
            with col1:
                pc = st.text_input('pus_cell')      
                
            with col2:
                sc = st.text_input('serum_creatinine')
                
            with col3:
                hemo = st.text_input('haemoglobin')
                
            with col1:
                wc = st.text_input('white_blood_cell_count')
                
            with col2:
                rc = st.text_input('red_blood_cell_count')
                
            
            # code for Prediction
            kidney_diagnosis = ''
            
            # creating a button for Prediction
            
            if st.button('kidney Disease Test Result'):
                kidney_prediction = kidney_model.predict([[age, bp , al, pc, bgr, sc, hemo,
                                                            wc, rc]])                          
                
                if (kidney_prediction[0] == 0):
                    kidney_diagnosis = 'The person does not has Chronic KIdney Disease'
                else:
                    kidney_diagnosis = 'The person has Chronic KIdney Disease'
                
            st.success(kidney_diagnosis)


        elif prediction_selected == 'Breast Cancer Prediction':
            # page title
            st.write("### Breast Cancer Prediction System")
            st.write("Predict the likelihood of breast cancer based on input features. Please provide the following information:")

            col1, col2, col3 = st.columns(3)
            
            with col1:
                radius_mean = st.text_input('radius_mean')
                
            with col2:
                texture_mean = st.text_input('texture_mean')
                
            with col3:
                perimeter_mean = st.text_input('perimeter_mean')
                
            with col1:
                area_mean  = st.text_input('area_mean')
                
            with col2:
                smoothness_mean = st.text_input('smoothness_mean')
                
            with col3:
                compactness_mean = st.text_input('compactness_mean')
                
            with col1:
                concavity_mean = st.text_input('concavity_mean')
                
            with col2:
                concave_points_mean = st.text_input('concave_points_mean')
                
            with col3:
                symmetry_mean = st.text_input('symmetry_mean')
                
            with col1:
                radius_se = st.text_input('radius_se')
                
            with col2:
                texture_se = st.text_input('texture_se')
                
            with col3:
                perimeter_se = st.text_input('perimeter_se')

            with col1:
                area_se = st.text_input('area_se')
                
            with col2:
                smoothness_se  = st.text_input('smoothness_se ')
                
            with col3:
                compactness_se = st.text_input('compactness_se')

            with col1:
                concavity_se = st.text_input('concavity_se')
                
            with col2:
                concave_points_se = st.text_input('concave points_se')
                
            with col3:
                symmetry_se = st.text_input('symmetry_se')
            
            with col1:
                fractal_dimension_se = st.text_input('fractal_dimension_se')
                
            with col2:
                radius_worst = st.text_input('radius_worst')
                
            with col3:
                texture_worst = st.text_input('texture_worst')

            with col1:
                perimeter_worst = st.text_input('perimeter_worst')
                
            with col2:
                area_worst  = st.text_input('area_worst')
                
            with col3:
                smoothness_worst = st.text_input('smoothness_worst')

            with col1:
                compactness_worst = st.text_input('compactness_worst')
                
            with col2:
                concavity_worst = st.text_input('concavity_worst')
                
            with col3:
                concave_points_worst = st.text_input('concave_points_worst')  

            with col1:
                symmetry_worst = st.text_input('symmetry_worst')
                
            with col2:
                fractal_dimension_worst = st.text_input('fractal_dimension_worst')          

            with col3:
                fractal_dimension_mean = st.text_input('fractal_dimension_mean')    
                
        
            # code for Prediction
            breast_diagnosis = ''   

        elif prediction_selected == 'Parkinson Disease Prediction':
            # page title
            st.write("### Parkinson's Disease Prediction System")
            st.write("Predict the likelihood of Parkinson's disease based on input features. Please provide the following information:")
            
            col1, col2, col3, col4, col5 = st.columns(5)  
            
            with col1:
                fo = st.text_input('MDVP:Fo(Hz)')
                
            with col2:
                fhi = st.text_input('MDVP:Fhi(Hz)')
                
            with col3:
                flo = st.text_input('MDVP:Flo(Hz)')
                
            with col4:
                Jitter_percent = st.text_input('MDVP:Jitter(%)')
                
            with col5:
                Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
                
            with col1:
                RAP = st.text_input('MDVP:RAP')
                
            with col2:
                PPQ = st.text_input('MDVP:PPQ')
                
            with col3:
                DDP = st.text_input('Jitter:DDP')
                
            with col4:
                Shimmer = st.text_input('MDVP:Shimmer')
                
            with col5:
                Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
                
            with col1:
                APQ3 = st.text_input('Shimmer:APQ3')
                
            with col2:
                APQ5 = st.text_input('Shimmer:APQ5')
                
            with col3:
                APQ = st.text_input('MDVP:APQ')
                
            with col4:
                DDA = st.text_input('Shimmer:DDA')
                
            with col5:
                NHR = st.text_input('NHR')
                
            with col1:
                HNR = st.text_input('HNR')
                
            with col2:
                RPDE = st.text_input('RPDE')
                
            with col3:
                DFA = st.text_input('DFA')
                
            with col4:
                spread1 = st.text_input('spread1')
                
            with col5:
                spread2 = st.text_input('spread2')
                
            with col1:
                D2 = st.text_input('D2')
                
            with col2:
                PPE = st.text_input('PPE')
                
            # code for Prediction
            parkinsons_diagnosis = ''
            
            # creating a button for Prediction    
            if st.button("Parkinson's Test Result"):
                parkinsons_prediction = parkinson_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,
                                                                APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
                
                if (parkinsons_prediction[0] == 1):
                    parkinsons_diagnosis = "The person has Parkinson's disease"
                else:
                    parkinsons_diagnosis = "The person does not have Parkinson's disease"
                
            st.success(parkinsons_diagnosis)
    
    elif selected == 'COVID-19 Prediction':
        # User Input
        st.title("COVID-19 Prediction App")
        st.write("Enter the details below to predict COVID-19.")

        # Organize input elements into three columns
        col1, col2, col3 = st.columns(3)

        with col1:
            Breathing_Problem = st.selectbox("Breathing Problem", [0, 1])
            Fever = st.selectbox("Fever", [0, 1])
            Dry_Cough = st.selectbox("Dry Cough", [0, 1])
            Sore_throat = st.selectbox("Sore Throat", [0, 1])

        with col2:
            Running_Nose = st.selectbox("Running Nose", [0, 1])
            Asthma = st.selectbox("Asthma", [0, 1])
            Chronic_Lung_Disease = st.selectbox("Chronic Lung Disease", [0, 1])
            Headache = st.selectbox("Headache", [0, 1])

        with col3:
            Heart_Disease = st.selectbox("Heart Disease", [0, 1])
            Diabetes = st.selectbox("Diabetes", [0, 1])
            Hyper_Tension = st.selectbox("Hyper Tension", [0, 1])
            Fatigue = st.selectbox("Fatigue", [0, 1])

        with col1:
            Gastrointestinal = st.selectbox("Gastrointestinal", [0, 1])  # Include this line
            Abroad_travel = st.selectbox("Abroad Travel", [0, 1])
            
        with col2:
            Family_working_in_Public_Exposed_Places = st.selectbox("Family working in Public Exposed Places", [0, 1])
            Visited_Public_Exposed_Places = st.selectbox("Visited Public Exposed Places", [0, 1])

        with col3:
            Contact_with_COVID_Patient = st.selectbox("Contact with COVID Patient", [0, 1])
            Attended_Large_Gathering = st.selectbox("Attended Large Gathering", [0, 1])
        
        # Predict Button
        if st.button("Predict"):
            # User Input DataFrame
            user_data = pd.DataFrame({
                'Breathing Problem': [Breathing_Problem],
                'Fever': [Fever],
                'Dry Cough': [Dry_Cough],
                'Sore Throat': [Sore_throat],
                'Running Nose': [Running_Nose],
                'Asthma': [Asthma],
                'Chronic Lung Disease': [Chronic_Lung_Disease],
                'Headache': [Headache],
                'Heart Disease': [Heart_Disease],
                'Diabetes': [Diabetes],
                'Hyper Tension': [Hyper_Tension],
                'Fatigue': [Fatigue],
                'Gastrointestinal': [Gastrointestinal],
                'Abroad Travel': [Abroad_travel],
                'Contact with COVID Patient': [Contact_with_COVID_Patient],
                'Attended Large Gathering': [Attended_Large_Gathering],
                'Visited Public Exposed Places': [Visited_Public_Exposed_Places],
                'Family working in Public Exposed Places': [Family_working_in_Public_Exposed_Places]
            })

            # Prediction
            prediction_text = predict_covid(user_data)

            # Display Prediction
            st.title("COVID-19 Prediction Result")
            st.write(f"The prediction is: {prediction_text}")
    
    elif selected == 'Endoscopic image Classification Classification':
        # st.title("Endoscopic image Classification Classification")
        # st.write("Upload an endoscopic image, and we'll classify it into one of the following categories:")
        # st.write("1. dyed-lifted-polyps\n2. dyed-resection-margins\n3. esophagitis\n4. normal-cecum")
        # st.write("5. normal-pylorus\n6. normal-z-line\n7. polyps\n8. ulcerative-colitis")
        home_page_content = """
        # Endoscopic Image Classification

        Dive into the world of endoscopic image classification with HealthVision AI. Our model can identify features like dyed-lifted-polyps, esophagitis, and more, aiding in efficient categorization of endoscopic images.

        ## How it Works:

        1. **Select 'Endoscopic Image Classification' from the sidebar.**
        2. **Upload your endoscopic images for analysis.**
        3. **Receive predictions about features identified in the images.**

        Explore the capabilities of our advanced machine learning model in categorizing endoscopic images. The results are meant for informational purposes and should be verified by medical professionals.

        """

        uploaded_file = st.file_uploader("Choose a file", type="jpg")
    
        if uploaded_file is not None:
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
            
            if st.button('Predict'):
                predicted_class, confidence = kvasir_predict_image(uploaded_file)
                st.success(f'Image Classification Prediction: {predicted_class} | Confidence: {confidence:.2f}%')


    elif selected == 'Diabetic Retinopathy Prediction':
        # st.title("Diabetic Retinopathy Detection")
        # st.write("Upload a retinal image, and we'll predict the presence of diabetic retinopathy.")
        home_page_content = """
        # Diabetic Retinopathy Prediction

        Predict the presence of diabetic retinopathy in retinal images with HealthVision AI. This prediction system offers a crucial tool for early detection and management of diabetic complications.

        ## How it Works:

        1. **Select 'Diabetic Retinopathy Prediction' from the sidebar.**
        2. **Upload retinal images for analysis.**
        3. **Receive predictions about the presence of diabetic retinopathy.**

        Stay proactive about your eye health with our diabetic retinopathy prediction system. Remember to consult with eye care professionals for accurate diagnosis and recommendations.

        """

        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            # Display the uploaded image and prediction
            image = Image.open(uploaded_file)

            # Make predictions
            prediction = diabetic_retinopathy_predict_class(image)

            # Display image and prediction side by side
            col1, col2 = st.columns(2)
            col1.image(image, caption="Uploaded Image.", use_column_width=True)
            col2.write("Prediction:")
            if prediction == 1:
                col2.write("No Diabetic Retinopathy")
            else:
                col2.write("Diabetic Retinopathy")


    elif selected == 'Alzheimer Brain Classification':
        st.title("Alzheimer Brain Classification")
        # # Additional information about the Alzheimer Brain Classification
        # st.write("This section utilizes a machine learning model trained to classify brain images for Alzheimer's disease.")
        # st.write("Upload a brain image, and the model will predict whether signs of Alzheimer's disease are present.")
        # st.write("The model was trained using state-of-the-art techniques to provide accurate predictions.")
        # st.write("Please note that predictions are based on the input image and may not replace professional medical advice.")
        home_page_content = """
        # Alzheimer's Brain Classification

        Upload brain images to our model, predicting the likelihood of Alzheimer's disease based on advanced machine learning techniques. Early detection can lead to better management and care.

        ## How it Works:

        1. **Select 'Alzheimer's Brain Classification' from the sidebar.**
        2. **Upload brain images for analysis.**
        3. **Receive predictions about the likelihood of Alzheimer's disease.**

        Utilize our advanced machine learning model for early insights into Alzheimer's disease. Consult with healthcare professionals for accurate diagnosis and recommendations.

        """

        # Adjust the size of the uploaded image
        uploaded_file = st.file_uploader("Choose an image...", type="jpg", key="1", accept_multiple_files=False)

        if uploaded_file is not None:
            # Display the uploaded image and prediction side by side
            col1, col2 = st.columns(2)

            # Display the uploaded image in the first column
            image = Image.open(uploaded_file)
            resized_image = image.resize((128, 128))  # Resize to match model input shape
            col1.image(resized_image, caption='Uploaded Image.', use_column_width=True)

            # Preprocess the image
            img_array = np.array(resized_image)
            img_array = keras.preprocessing.image.img_to_array(img_array)
            img_array = keras.applications.efficientnet.preprocess_input(img_array)  # Assuming EfficientNet preprocessing

            img_array = np.expand_dims(img_array, axis=0)  # Create a batch

            # Make predictions
            predictions = alzheimer_model.predict(img_array)
            score = np.argmax(predictions[0])

            # Display the predicted class and confidence in the second column
            col2.subheader(f"Prediction: {alzheimer_class_names[score]}")
            col2.write(f"Confidence: {100 * np.max(predictions[0]):.2f}%")


    elif selected == 'Cell Segmentation':
        # st.write("## Cell Segmentation Task:")
        # st.write("Upload an image, and we'll perform cell segmentation to highlight cell boundaries.")
        home_page_content = """
        # Cell Segmentation

        Visualize the intricate world of cells with HealthVision AI's segmentation model. Upload medical images, and witness the delineation of cell boundaries for in-depth analysis.

        ## How it Works:

        1. **Select 'Cell Segmentation' from the sidebar.**
        2. **Upload medical images for segmentation.**
        3. **View the segmented cell boundaries for analysis.**

        Explore the details of cell structures with our advanced segmentation model. The results provide a visual representation and are intended for informational purposes.

        """

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
    
    st.markdown(home_page_content)


if __name__ == "__main__":
    main()