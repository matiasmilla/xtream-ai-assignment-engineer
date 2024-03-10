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

encoder = joblib.load('../encoder.sav')
scaler = joblib.load('../scaler.sav')
model = XGBRegressor()
model.load_model('../diamonds_model.json')

class Features(BaseModel):
    x: float
    y: float
    z: float
    color: str
    clarity: str

class FeaturesList(BaseModel):
    data: list

@app.post('/predict_one')
def predict_one(json_data: Features):
    parameters = {
        'size': [json_data.x*json_data.y*json_data.z],
        'color': [json_data.color],
        'clarity': [json_data.clarity]
    }
    df = pd.DataFrame.from_dict(parameters)
    df[['color', 'clarity']] = encoder.transform(df[['color', 'clarity']])
    df = scaler.transform(df)
    prediction = model.predict(df)
    return {'price': int(prediction[0])}

@app.post('/predict_many')
def predict_many(json_data: FeaturesList):
    df = pd.DataFrame(json_data.data)
    df.set_index('id', inplace=True)
    df['size'] = df['x']*df['y']*df['z']
    df.drop(['x', 'y', 'z'], axis=1, inplace=True)
    df[['color', 'clarity']] = encoder.transform(df[['color', 'clarity']])
    # reorder features for model
    df = df[['size', 'color', 'clarity']]
    features = scaler.transform(df)
    predictions = model.predict(features)
    int_prices = list(map(int, predictions))
    results = dict(zip(df.index, int_prices))
    return {'results': results}

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
