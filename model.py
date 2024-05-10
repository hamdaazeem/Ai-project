import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load your models (make sure the paths are correct)
logistic_model = joblib.load('log.pkl')
random_forest_model = joblib.load('Rm.pkl')
ann_model = load_model('ann.h5')


# Define a function to preprocess input data
def preprocess_data(input_df):
    # Replace infinite values with NaN
    input_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Standardize numerical features
    scaler = StandardScaler()
    numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    input_df[numerical_features] = scaler.fit_transform(input_df[numerical_features])

    return input_df

def predict(input_data):
    # Preprocess data
    input_data = preprocess_data(input_data)

    # Make predictions with each model
    logistic_pred = logistic_model.predict(input_data)
    rf_pred = random_forest_model.predict(input_data)
    input_data = np.array(input_data)
    sc = StandardScaler()
    input_data = sc.fit_transform(input_data)
    ann_pred = (ann_model.predict(input_data) > 0.5).astype("int32")
    return logistic_pred, rf_pred, ann_pred

# Streamlit interface
st.title('Heart Disease Prediction')
st.write("Enter patient details:")

# Input fields based on the dataset
age = st.number_input('Age', min_value=18, max_value=100, value=50)
sex = st.selectbox('Sex', ['M', 'F'])
chest_pain_type = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'])
resting_bp = st.number_input('Resting Blood Pressure', min_value=80, max_value=200, value=120)
cholesterol = st.number_input('Cholesterol', min_value=100, max_value=400, value=200)
fasting_bs = st.radio('Fasting Blood Sugar > 120 mg/dl', [0, 1])
resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
max_hr = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=200, value=150)
exercise_angina = st.radio('Exercise Induced Angina', ['Y', 'N'])
oldpeak = st.number_input('Oldpeak', min_value=0.0, max_value=10.0, value=0.0, step=0.1, format="%.1f")
st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

# Create DataFrame from input
input_df = pd.DataFrame([[age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, 
                          resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]],
                        columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                                 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'])

df = pd.read_csv("heart.csv")
data = pd.get_dummies(df, columns=['Sex','ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])
X_encoded = pd.get_dummies(data.drop('HeartDisease', axis=1))

New_data_encoded = pd.get_dummies(input_df, columns=['Sex','ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])
# Adding missing dummy variables with 0s
missing_cols = set(X_encoded.columns) - set(New_data_encoded.columns)
for c in missing_cols:
    New_data_encoded[c] = 0

# Ensuring the order of columns matches the training data
New_data_encoded = New_data_encoded[X_encoded.columns]

classes = ['Heart disease not found', 'Heart disease found']
if st.button('Predict'):
    New_X_encoded = preprocess_data(New_data_encoded)
    results = predict(New_X_encoded)
    st.write('Predictions:')
    #st.write(f'Logistic Regression: {classes[results[0][0]]}')  # Adjusted for possible array output
    st.write(f'Random Forest: {classes[results[1][0]]}')
   # st.write(f'ANN: {results[2][0]}')
