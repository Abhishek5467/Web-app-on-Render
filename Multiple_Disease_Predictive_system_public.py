# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 11:10:13 2024

@author: samay
"""

import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu


# loading the saved models

Heart_disease_model = pickle.load(open("Heart_disease_model.sav", "rb"))

Diabetes_model = pickle.load(open("Diabetes_model.sav", "rb"))


def Heart_disease_prediction(input_data):
    
    # change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array as we are prediction for only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = Heart_disease_model.predict(input_data_reshaped)
    # print(prediction)
    

    if (prediction[0] == 0):
      return 'The Person does not have a Heart Disease!'
    else:
      return 'The Person has Heart Disease!'


def Diabetes_prediction(input_data):
    
    # change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    # reshape the numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    prediction = Diabetes_model.predict(input_data_reshaped)
    print(hello)

    # printing the prediction
    if (prediction[0] == 0):
      return 'The Person does not have a Diabetes!'
    else:
      return 'The Person has Diabetes!'    
    

# sidebar for navigate

with st.sidebar:
    
    selected = option_menu("Multiple Disease Predictive System",
                           ['Heart disease prediction',
                            'Diabetes prediction'],
                           icons = ["heart-pulse-fill", "file-medical"],
                           default_index = 0)
    

# Heart disease prediction page
if(selected == "Heart disease prediction"):
    
    # page title
    st.title("Heart Disease Prediction")
    
    # getting the input data from user
    
    # columns for input fields
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age")    
        
    with col2:
        sex = st.number_input("sex(1 for male & 0 for female")
        
    with col3:
        chest_pain_type = st.number_input("chest pain type (0 for Typical angina, 1 for Non-anginal pain, 2 for Atypical angina & 3 for Asymptomatic)")
        
    with col1:
        resting_blood_pressure = st.number_input("resting blood pressure")
        
    with col2:
        cholestoral = st.number_input("cholestoral")
        
    with col3:
        fasting_blood_sugar = st.number_input("fasting blood sugar(0 for lower than 120 mg/ml & 1 for gretaer than 120 mg/ml)")    
    
    with col1:
        rest_ecg = st.number_input("rest ecg(0 for Normal, 1 for ST-T wave abnormality & 2 for Left ventricular hypertrophy)")
        
    with col2:
        Max_heart_rate = st.number_input("Max heart rate")
        
    with col3:
        exercise_induced_angina = st.number_input("exercise_induced_angina(0 for No & 1 for Yes)")
    
    with col1:
        oldpeak = st.number_input("oldpeak")
        
    with col2:
        slope = st.number_input("slope(0 for Downslopping, 1 for Upsloping & 2 for Flat)")
        
    with col2:
        vessels_colored_by_flourosopy = st.number_input("vessels colored by flourosopy")
        
    with col1:
        thalassemia = st.number_input("thalassemia(0 for Normal, 1 for Fixed Defect, 2 for Reversable Defect, 3 for No)")
    
    
    
    # code for prediction
    Heart_diagnosis = ""
    
    # creating button for prediction
    if(st.button("Heart Disease Prediction Result")):
        Heart_prediction = Heart_disease_model.predict(np.asarray([age,sex,chest_pain_type,resting_blood_pressure,cholestoral,fasting_blood_sugar,rest_ecg,Max_heart_rate,exercise_induced_angina,oldpeak,slope,vessels_colored_by_flourosopy,thalassemia]).reshape(1,-1))
        
        if (Heart_prediction[0] == 0):
          Heart_diagnosis = 'The Person does not have a Heart Disease!'
        else:
          Heart_diagnosis = 'The Person has Heart Disease!'
        
    st.success(Heart_diagnosis)
    
else:
    
    # page title
    st.title("Diabetes Prediction")
    
    
    # getting the input data from user
    
    # columns for input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.number_input("Pregnancies")
    
    with col2:
        Glucose = st.number_input("Glucose")
     
    with col3:
        BloodPressure = st.number_input("Blood Pressure")
        
    with col1:
        SkinThickness = st.number_input("Skin Thickness")
    
    with col2:
        Insulin = st.number_input("Insulin")
     
    with col3:
        BMI = st.number_input("BMI")

    with col1:
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function")
    
    with col2:
        Age = st.number_input("Age")


    
    # code for prediction
    diabetes_diagnosis = ""
    
    # creating button for prediction
    if(st.button("Diabetes Prediction Result")):
        diabetes_prediction = Diabetes_model.predict(np.asarray([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]).reshape(1,-1))
        
        if (diabetes_prediction[0] == 0):
          diabetes_diagnosis = 'The Person does not have a Diabetes!'
        else:
          diabetes_diagnosis = 'The Person has Diabetes!'
        
    st.success(diabetes_diagnosis)
