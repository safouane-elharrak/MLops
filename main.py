# from main import FastAPI
import pickle
import uvicorn
import asyncio
import numpy as np 
import pandas as pd
import nest_asyncio
from model import train, predict, convert

app = FastAPI(debug=True)
@app.get('/')
def home():
    return {'text':'welcome home'}

@app.get('/predict')
async def predict(Id:int):
    df_test = pd.read_csv('C:/Users/Safouane Elh/Documents/MBD S3/Deep Learning/MLOPS/test.csv')
    loaded_model = pickle.load(open('C:/Users/Safouane Elh/Documents/MBD S3/Deep Learning/MLOPS/lstm_model.pkl', 'rb'))
  # creating submission file 
    xtest = np.load('C:/Users/Safouane Elh/Documents/MBD S3/Deep Learning/MLOPS/parrot.npy')
    submission_file = loaded_model.predict(xtest)
  # we will keep every value between 0 and 20
    submission_file = submission_file.clip(0,20)
  # creating dataframe with required columns 
    submission_trp = pd.DataFrame({'ID':df_test['ID'],'item_cnt_month':submission_file.ravel()})
    val = submission_trp['col_name'].values[Id]
      
    return {'number of products sold':val}


if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app)

# import pickle

# loaded_model = pickle.load(open('C:/Users/Safouane Elh/Documents/MBD S3/Deep Learning/MLOPS/lstm_model.pkl', 'rb'))
# import keras
# print(keras.__version__)