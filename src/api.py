from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import logging
from src.model import convert_bool_to_numeric  # Importing the function

app = FastAPI()

logging.basicConfig(level=logging.DEBUG)

class DataRow(BaseModel):
    data: dict

def clean_data(data):
    """
    Cleans and preprocesses the data.
    """
    data['x12'] = data['x12'].str.replace('$', '').str.replace(',', '').str.replace(')', '').str.replace('(', '-')
    data['x12'] = pd.to_numeric(data['x12'], errors='coerce')
    data['x63'] = data['x63'].str.replace('%', '')
    data['x63'] = pd.to_numeric(data['x63'], errors='coerce')
    return data

def create_dummies(data, base_data, dummy_columns):
    """
    Creates dummy variables for categorical columns.
    """
    for col in ['x5', 'x31', 'x81', 'x82']:
        dummies = pd.get_dummies(data[col], drop_first=True, prefix=col, prefix_sep='_', dummy_na=True)
        base_data = pd.concat([base_data, dummies], axis=1, sort=False)
    
    # Ensure all expected dummy columns are present
    for col in dummy_columns:
        if col not in base_data.columns:
            base_data[col] = 0
    
    return base_data

@app.post("/predict")
def predict(data: DataRow):
    try:
        input_data = pd.DataFrame([data.data])
        model_path = 'model.pkl'
        variables_path = 'variables.pkl'

        if not os.path.exists(model_path) or not os.path.exists(variables_path):
            raise FileNotFoundError(f"Model file or variables file not found. Model path: {model_path}, Variables path: {variables_path}")

        logging.debug(f"Loading model from {model_path}")
        model = joblib.load(model_path)

        logging.debug(f"Loading variables from {variables_path}")
        variables = joblib.load(variables_path)

        logging.debug(f"Input data before cleaning: {input_data}")
        
        # Clean the input data
        input_data = clean_data(input_data)
        logging.debug(f"Input data after cleaning: {input_data}")

        # Initialize standardized data (e.g., scaling if necessary)
        input_data_std = input_data.copy()  # Assuming standardization is done similarly to training data

        # Create dummies for categorical variables
        input_data_std = create_dummies(input_data, input_data_std, variables)
        logging.debug(f"Input data after creating dummies: {input_data_std}")

        # Convert boolean columns to numeric
        input_data_std = convert_bool_to_numeric(input_data_std, variables)
        logging.debug(f"Input data after converting booleans: {input_data_std}")

        # Ensure all expected columns are present
        missing_columns = [col for col in variables if col not in input_data_std.columns]
        if missing_columns:
            logging.error(f"Missing columns: {missing_columns}")
            raise ValueError(f"Missing columns: {missing_columns}")

        input_data_std = input_data_std[variables]
        logging.debug(f"Input data before prediction: {input_data_std}")
        logging.debug(f"Data types of input data: {input_data_std.dtypes}")

        # Making prediction
        prob = model.predict(input_data_std)
        predicted_class = (prob > 0.5).astype(int)
        print(predicted_class)
        print(prob)
        
        return {
            "predicted_class": predicted_class.tolist(),
            "probability": prob.tolist(),
            "model_features": variables
        }
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
