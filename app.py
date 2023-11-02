import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('money.h5')

# Define the function for making predictions
def make_predictions(new_data):
    # Encode new_data
    new_data_encoded = pd.get_dummies(new_data['Currency'], drop_first=True)
    
    # Expand input shape to (None, 168)
    new_data_expanded = pd.DataFrame(np.zeros((new_data_encoded.shape[0], 168 - new_data_encoded.shape[1])), columns=[f'Currency_{i}' for i in range(new_data_encoded.shape[1], 168)])
    new_data_encoded = pd.concat([new_data_encoded, new_data_expanded], axis=1)
    
    # Make predictions
    predictions = model.predict(new_data_encoded)
    
    return predictions

# Streamlit app
st.title('Exchange Rate Prediction App')
st.sidebar.header('Input Parameters')

# Sidebar inputs for user to enter new data
currency = st.sidebar.text_input('Currency (e.g., EUR, GBP)', 'EUR')
new_data = pd.DataFrame({'Currency': [currency]})

# Predict button
if st.sidebar.button('Predict'):
    predictions = make_predictions(new_data)
    st.subheader('Predicted Exchange Rate')
    st.write(predictions)

