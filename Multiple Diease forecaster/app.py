# to run "streamlit run app.py"
import streamlit as st
import pickle 
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


diabetic_model = pickle.load(open('model.pkl','rb')) 
heart_disease_model = pickle.load(open('heart.pkl','rb'))
kidney_model = pickle.load(open('kidney.pkl','rb'))
breast_cancer_model = pickle.load(open('Breast_cancer.pkl','rb'))
parkinson_model = pickle.load(open('parkinsons.pkl','rb'))



with st.sidebar:
    
    selected = option_menu('Prediction System',
                          
                          ['Diabetes Prediction',
                          'Chronic Kidney Disease',
                          'Breast Cancer Prediction',
                          'Heart Disease Prediction',
                          'parkinson Disease Prediction'],
                           default_index=0)



################## Diabetes Prediction ###########################

if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction')
    
    
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




################### Chronic KIdney Disease  #####################

if (selected == 'Chronic Kidney Disease'):
    
    # page title
    st.title('Chronic Kidney Disease prediction')
    
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



####################### Heart Disease Prediction Page ########################

if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 0):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)      




########################  Breast_Cancer_Prediction #####################

if (selected == 'Breast Cancer Prediction'):
    
    # page title
    st.title('Breast Cancer Prediction')
    
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
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        breast_prediction = breast_cancer_model.predict([[radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,
                                                        compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,
                                                        fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,
                                                        smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,
                                                        fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,
                                                        smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,
                                                        symmetry_worst,fractal_dimension_worst]])                          
        
        if (breast_prediction[0] == 0):
          breast_diagnosis = 'Benign (not cancerous)'
        else:
          breast_diagnosis = 'Malignant (cancerous)'
        
    st.success(breast_diagnosis)  



##############  Parkinsons Disease #######################

if (selected == "parkinson Disease Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
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




         