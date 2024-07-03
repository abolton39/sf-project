from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.impute import SimpleImputer
from src.data_prep import add_missing_columns, clean_data, create_dummies, convert_bool_to_numeric

app = FastAPI()

logging.basicConfig(level=logging.DEBUG)

class PredictionRequest(BaseModel):
    data: dict

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        input_data = pd.DataFrame([request.data])
        model_path = '/app/models/model.pkl'
        variables_path = '/app/models/variables.pkl'
        scaler_path = '/app/models/scaler.pkl'

        # Load all needed .pkl files
        logging.debug(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        logging.debug(f"Loading variables from {variables_path}")
        variables = joblib.load(variables_path)
        logging.debug(f"Loading model from {scaler_path}")
        scaler = joblib.load(scaler_path)

        logging.debug(f"Input data before cleaning: {input_data}")
        
        # Clean the input data
        input_data = clean_data(input_data)
        logging.debug(f"Input data after cleaning: {input_data}")

        # Mean imputation for empty values
        try:
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            data_imputed = pd.DataFrame(imputer.fit_transform(input_data.drop(columns=['x5', 'x31', 'x81', 'x82'])), 
                                             columns=input_data.drop(columns=['x5', 'x31', 'x81', 'x82']).columns)
        except ValueError as e:
            # For single row calls, mean doesn't work. Fill "" and NAN with 0s
            input_data.replace("", np.nan, inplace=True)
            data_imputed = input_data.drop(columns=['x5', 'x31', 'x81', 'x82']).fillna(0)
        logging.debug(f"Input data after imputation: {data_imputed}")

        # Scale the data
        data_std = pd.DataFrame(scaler.transform(data_imputed))
        logging.debug(f"Input data after scaling: {data_std}")

        # Create dummies for non-numeric variables
        data_dummies = create_dummies(input_data, data_std)
        logging.debug(f"Input data after creating dummies: {data_dummies}")

        # Add any missing columns. Only needed when not all columns are present for 1 hot
        data_all = add_missing_columns(data_dummies, variables)
        data_all = data_all[variables]

        # Convert boolean columns to numeric
        data_all = convert_bool_to_numeric(data_all, variables)
        logging.debug(f"Input data before prediction: {data_all}")
        logging.debug(f"Data types of input data: {data_all.dtypes}")

        # Prediction
        prob = model.predict(data_all)
        # Specify threshold here, could be passed as a variable by customer
        prob_threshold = 0.75
        predicted_class = (prob > prob_threshold).astype(int)
        
        return {
            "business_outcome": predicted_class.tolist(),
            "phat": prob.tolist(),
            "model_features": variables
        }
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
