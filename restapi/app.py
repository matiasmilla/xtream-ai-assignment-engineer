from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from io import StringIO
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
async def predict_one(json_data: Features):
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
async def predict_many(json_data: FeaturesList):
    df = pd.DataFrame(json_data.data)
    df.set_index('id', inplace=True)
    int_prices = process_data(df)
    results = dict(zip(df.index, int_prices))
    return {'results': results}

@app.post('/predict_many_csv')
async def predict_many_csv(file: UploadFile = File(...)):
    if file.content_type != 'text/csv':
        return {"error": "input must be a csv file"}
    content = await file.read()
    content = content.decode()
    df = pd.read_csv(StringIO(content))
    prices = process_data(df.copy())
    df['price'] = prices
    stream = StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type='text/csv')
    response.headers['Content-Disposition'] = 'attachment; filename=prices.csv'
    return response

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
