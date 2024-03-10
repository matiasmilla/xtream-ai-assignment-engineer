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

def process_data(df):
    df['size'] = df['x']*df['y']*df['z']
    df.drop(['x', 'y', 'z'], axis=1, inplace=True)
    df[['color', 'clarity']] = encoder.transform(df[['color', 'clarity']])
    # reorder features for model
    df = df[['size', 'color', 'clarity']]
    features = scaler.transform(df)
    predictions = model.predict(features)
    int_prices = list(map(int, predictions))
    return int_prices

@app.post('/predict_one')
def predict_one(json_data: Features):
    parameters = {
        'x': [json_data.x],
        'y': [json_data.y],
        'z': [json_data.z],
        'color': [json_data.color],
        'clarity': [json_data.clarity]
    }
    df = pd.DataFrame.from_dict(parameters)
    price = process_data(df)[0]
    return {'price': price}

@app.post('/predict_many')
def predict_many(json_data: FeaturesList):
    df = pd.DataFrame(json_data.data)
    df.set_index('id', inplace=True)
    int_prices = process_data(df)
    results = dict(zip(df.index, int_prices))
    return {'results': results}

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
