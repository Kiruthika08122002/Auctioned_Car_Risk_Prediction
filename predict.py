import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load files
with open('risk_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Input data (sample)
input_data = {
    'Car_Name': ['Maruti'],          # Use known car name
    'Year': [2015],
    'Selling_Price': [3.5],
    'Present_Price': [5.0],
    'Kms_Driven': [50000],
    'Fuel_Type': ['Petrol'],
    'Seller_Type': ['Dealer'],
    'Transmission': ['Manual'],
    'Owner': [0]
}

input_df = pd.DataFrame(input_data)

# Apply label encoding with unknown label handling
for col in input_df.columns:
    if col in label_encoders:
        le = label_encoders[col]
        try:
            input_df[col] = le.transform(input_df[col])
        except ValueError:
            print(f"Unknown value found in column '{col}'. Replacing with default (first known label).")
            input_df[col] = [le.transform([le.classes_[0]])[0]]

# Scale the input (without Risk column)
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)
print("Predicted Risk Level:", prediction[0])