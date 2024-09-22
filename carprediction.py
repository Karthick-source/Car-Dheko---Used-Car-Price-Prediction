import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model
try:
    with open('C:\\Users\\admin\\python\\Car Dheko - Used Car Price Prediction\\random_forest.joblib', 'rb') as file:
        model = joblib.load(file)
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Check if the loaded model is correct
if not hasattr(model, 'predict'):
    st.error("The loaded model is not valid. Please ensure that the correct machine learning model is loaded.")
else:
    # Title of the application
    st.title("Car Price Prediction App")

    # Instructions for the user
    st.write("### Input the following car features to predict the price:")

    # Car feature inputs
    model_year = st.number_input('Model Year', min_value=1990, max_value=2024, value=2015)
    bt = st.selectbox('Body Type', ['Hatchback', 'Sedan', 'SUV', 'Convertible', 'Coupe', 'Other'])
    mileage = st.number_input('Mileage (in km)', min_value=0, max_value=500000, value=10000)
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])

    model_name = st.selectbox('Model Name', ['Maruti Celerio', 'Ford Ecosport', 'Tata Tiago', 'Hyundai Xcent',
                                              'Maruti SX4 S Cross', 'Jeep Compass', 'Datsun GO', 'Hyundai Venue','Hyundai i20',
'Audi A6','Maruti 800','Maruti Alto 800','Volkswagen Polo'])

    variant_name = st.selectbox('Variant Name', ['VXI', '1.5 Petrol Titanium BSIV', '1.2 Revotron XZ', 
                                                  'XZA Plus P Dark Edition AMT', 'X-Line DCT', 'C 200 CGI Elegance','DieselRXZ', '110PS AMT BSIV', 'Sportz 1.2','Asta Option 1.4 CRDi',
'1.2 Zeta','SX','35 TDI','VXI Plus BSVI','SLE 7S BSIII','Cooper S BSVI','XZ Plus','GTX Plus DCT'])

    ft = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'Electric', 'Hybrid'])
    km = st.number_input('Kilometers Driven', min_value=0, max_value=500000, value=50000)

    # Encoding categorical features manually based on the dataset
    bt_mapping = {'Hatchback': 2, 'Sedan': 7, 'SUV': 8, 'Convertible': 3, 'Coupe': 1, 'Other': 4}
    transmission_mapping = {'Manual': 1, 'Automatic': 0}
    ft_mapping = {'Petrol': 4, 'Diesel': 1, 'Electric': 2, 'Hybrid': 3}

    # Unique model and variant names for encoding
    unique_model_names = ['Maruti Celerio', 'Ford Ecosport', 'Tata Tiago', 'Hyundai Xcent',
                          'Maruti SX4 S Cross', 'Jeep Compass', 'Datsun GO', 'Hyundai Venue']
    
    unique_variant_names = ['VXI', '1.5 Petrol Titanium BSIV', '1.2 Revotron XZ', 
                            'XZA Plus P Dark Edition AMT', 'X-Line DCT', 'C 200 CGI Elegance']

    model_name_mapping = {name: i for i, name in enumerate(unique_model_names)}
    variant_name_mapping = {name: i for i, name in enumerate(unique_variant_names)}

    # Normalize the inputs
    model_year_normalized = (model_year - 1990) / (2024 - 1990)  # Scale between 1990 and 2024
    mileage_normalized = mileage / 500000  # Assume 500000 is the max mileage
    km_normalized = km / 500000  # Same scaling for kilometers driven

    # Prepare the input for prediction using the selected features
    input_data = pd.DataFrame({
        'modelYear': [model_year_normalized],
        'bt': [bt_mapping[bt]],
        'mileage': [mileage_normalized],
        'transmission': [transmission_mapping[transmission]],
        'model': [model_name_mapping.get(model_name, -1)],  # Use -1 for unknown models
        'variantName': [variant_name_mapping.get(variant_name, -1)],  # Use -1 for unknown variants
        'ft': [ft_mapping[ft]],
        'km': [km_normalized]
    })

    # Predict price based on the user input
    if st.button('Predict Price'):
        try:
            prediction = model.predict(input_data)
            st.success(f"The predicted price of the car is: ₹{prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

# Error handling and success message
st.write("#### Ensure you have inputted correct details. The model uses regression based on historical data to predict prices.")
