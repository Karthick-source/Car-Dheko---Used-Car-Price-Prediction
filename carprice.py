import streamlit as st
import joblib
import pandas as pd
import pickle
import numpy as np

# Load the saved model, label encoders, and scalers
model = joblib.load('gradient_model.joblib')

with open('label_encoder.pkl', 'rb') as f:
    L_encoder = pickle.load(f)

with open('price.pkl', 'rb') as r:
    price_scaler = pickle.load(r)  # MinMaxScaler for price

# Function to validate the model
def validate_model(model):
    if not hasattr(model, 'predict'):
        st.error("Invalid model. Please load the correct machine learning model.")
        return False
    return True

# Input fields for car features
model_year = st.number_input('modelYear', min_value=1990, max_value=2024, value=2015)
body_type = st.selectbox('Body_type', ['Hatchback', 'Sedan', 'SUV', 'Convertible', 'Coupe', 'Other'])
mileage = st.number_input('mileage (km/l)', min_value=5.0, max_value=50.0, step=0.1, value=15.5)
transmission = st.selectbox('transmission', ['Manual', 'Automatic'])
model_name = st.selectbox('model', [
    'Maruti Celerio', 'Ford Ecosport', 'Tata Tiago', 'Hyundai Xcent',
    'Maruti SX4 S Cross', 'Jeep Compass', 'Datsun GO', 'Hyundai Venue',
    'Audi A6', 'Maruti 800', 'Volkswagen Polo'
])
variant_name = st.selectbox('variantName', [
    'VXI', '1.5 Petrol Titanium BSIV', '1.2 Revotron XZ',
    'XZA Plus P Dark Edition AMT', 'X-Line DCT', 'C 200 CGI Elegance'
])
fuel_type = st.selectbox('Fuel_type', ['Petrol', 'Diesel', 'Electric', 'Hybrid'])
engine_displacement = st.number_input('Engine_displacement', min_value=0, max_value=5000, step=1, value=500)
kilometers_driven = st.number_input('Kilometers', min_value=0, max_value=500000, step=1000, value=50000)
engine_type = st.selectbox('Engine_type', [
    'K10B Engine', 'Ti-VCT Petrol Engine', 'Revotron Engine',
    'Kappa VTVT Petrol Engine', 'DDiS 200 Diesel Engine',
    'K Series VVT Engine', 'K10C'
])
central_variant = st.number_input('centralVariantId', min_value=0, max_value=15000, step=1, value=5000)
city = st.selectbox('City', ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Jaipur', 'Kolkata'])
seats = st.number_input('seats', min_value=2, max_value=10, step=1, value=5)
ownerNo = st.number_input('ownerNo', min_value=1, max_value=10, step=1, value=1)
insurance_validity = st.selectbox('Insurance Validity', [
    'Comprehensive', 'Third Party insurance', 'First Party insurance'
])

# Create feature input data for prediction
numerical_features = pd.DataFrame({
    'modelYear': [model_year],
    'Kilometers': [kilometers_driven],
    'ownerNo': [ownerNo],
    'seats': [seats],
    'mileage': [mileage],
    'centralVariantId': [central_variant],
 'Engine_displacement': [engine_displacement],
})

# Categorical input data
categorical_data = pd.DataFrame({
    'Body_type': [body_type],
    'transmission': [transmission],
    'model': [model_name],
    'variantName': [variant_name],
    'Fuel_type': [fuel_type],
    'Insurance Validity': [insurance_validity],
    'Engine_type': [engine_type],
    'City': [city]
})

# Transform categorical features using LabelEncoders
for column in categorical_data.columns:
    categorical_data[column] = L_encoder[column].transform(categorical_data[column])

# Combine the numerical and encoded categorical features
input_data = pd.concat([numerical_features.reset_index(drop=True),
                        categorical_data.reset_index(drop=True)], axis=1)

# Ensure the input data matches the model’s expected feature order
expected_feature_names = [
    'modelYear','Body_type', 'Engine_displacement', 'transmission', 'centralVariantId',
    'mileage','Engine_type','variantName',  'City', 'model','Kilometers',
    'seats', 'Insurance Validity','ownerNo','Fuel_type'
]

input_data = input_data[expected_feature_names]

# Validate the model and make predictions
if st.button('Calculate Price'):
    if validate_model(model):
        with st.spinner('Predicting price...'):
            # Predict the scaled price
            try:
                predicted_scaled = model.predict(input_data)

                # Reshape to (1, -1) if necessary for scaler
                predicted_scaled = np.array(predicted_scaled).reshape(-1, 1)

                # Inverse-transform the scaled prediction to original price
                predicted_price = price_scaler.inverse_transform(predicted_scaled)

                st.success(f"The predicted price is: ₹ {predicted_price[0][0]:,.2f} lakh")
            except ValueError as e:
                st.error(f"Prediction failed: {e}")
