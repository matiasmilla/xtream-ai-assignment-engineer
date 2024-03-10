from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sklearn.preprocessing import TargetEncoder, StandardScaler
import json
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import uvicorn

app = FastAPI()

scaler = joblib.load('../scaler.sav')
model = XGBRegressor()
model.load_model('../diamonds_model.json')

class Features(BaseModel):
    x: float
    y: float
    z: float
    color: float
    clarity: float

@app.post('/predict')
def predict(json_data: Features):
    size = [json_data.x*json_data.y*json_data.z]
    color = [json_data.color]
    clarity = [json_data.clarity]
    parameters = {
        'size': size,
        'color': color,
        'clarity': clarity
    }
    df = pd.DataFrame.from_dict(parameters)
    df = scaler.transform(df)
    prediction = model.predict(df)
    return {'price': int(prediction[0])}

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=1984)
