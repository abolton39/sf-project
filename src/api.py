from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI()

class DataRow(BaseModel):
    data: dict

@app.post("/predict")
def predict(data: DataRow):
    try:
        input_data = pd.DataFrame([data.data])
        model_path = 'model.pkl'
        variables_path = 'variables.pkl'

        if not os.path.exists(model_path) or not os.path.exists(variables_path):
            raise FileNotFoundError(f"Model file or variables file not found. Model path: {model_path}, Variables path: {variables_path}")

        model = joblib.load(model_path)
        variables = joblib.load(variables_path)
        input_data = input_data[variables]
        prob = model.predict_proba(input_data)[:, 1]
        predicted_class = (prob > 0.5).astype(int)
        return {"predicted_class": predicted_class.tolist(), "probability": prob.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
