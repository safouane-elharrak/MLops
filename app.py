from fastapi import FastAPI
import pickle
import uvicorn
import asyncio
import numpy as np 
import pandas as pd


app = FastAPI(debug=True)
@app.get('/')
def home():
  return {'text':'welcome home'}


@app.get('/predict')
def predict(Id:int):
    model = pd.read_csv(open("C:/Users/Safouane Elh/Documents/MBD S3/Deep Learning/MLOPS/submission.csv","rb"))
    val = model['item_cnt_month'].values[Id]
    val2 = model['shop_id'].values[Id]
    val3 = model['item_id'].values[Id]
    return {'the number is ':val}
async def main(Id:int):
    model = pd.read_csv(open("C:/Users/Safouane Elh/Documents/MBD S3/Deep Learning/MLOPS/submission.csv","rb"))
    val = model['item_cnt_month'].values[Id]
    val2 = model['shop_id'].values[Id]
    val3 = model['item_id'].values[Id]
    return {'number of products sold' ,val,'shop_id',val2,'item_id',val3}
import nest_asyncio

if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app)
