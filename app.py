# !pip install flask-ngrok

from flask import Flask,request, url_for, redirect, render_template

from flask_ngrok import run_with_ngrok
import numpy as np
import pandas as pd
from Sentiment_predict import sentiment_predict
from Sentiment_predict import Tokenization
import pre_processing

from tensorflow.keras import models



TRAINED_MODEL_PATH_H5 = '/content/drive/MyDrive/Final year project/trained_main_model_v2.h5'
EMBEDDING_MATRIX_PATH = '/content/drive/MyDrive/Final year project/Datasets/Embedding/main_embedding_matrix_pickle'

trained_model = models.load_model(TRAINED_MODEL_PATH_H5)
tokenizer = Tokenization(EMBEDDING_MATRIX_PATH)
# model=pickle.load(open('model.pkl','rb'))





# TRAINED_MODEL_PATH_H5 = '/content/drive/MyDrive/Final year project/trained_main_model_v2.h5'
# EMBEDDING_MATRIX_PATH = '/content/drive/MyDrive/Final year project/Datasets/Embedding/main_embedding_matrix_pickle'

# trained_model = models.load_model(TRAINED_MODEL_PATH_H5)
# tokenizer = Tokenization(EMBEDDING_MATRIX_PATH)
# # model=pickle.load(open('model.pkl','rb'))


app = Flask(__name__)
run_with_ngrok(app) 

@app.route('/')
def hello_world():
    return render_template("index_v2.html",display_meter='None')


@app.route('/predict',methods=['POST','GET'])
def predict():
    input_text=[x for x in request.form.values()]
    print(input_text)
    prediction,probability,_= sentiment_predict(trained_model,[pre_processing.pre_processing(input_text[0])],tokenizer)
    # prediction = 'SIMIELE'
    print(probability)
    output=prediction
    print(prediction)
    if output>=5:
      inPercent = probability[0]*prediction
      print(inPercent)
      output = int(inPercent)
      print(output)
      # return render_template('index_v2.html',pred='{}'.format('positive'),score=inPercent*10)
      if output==5:
        return render_template('index_v2.html',pred='{}'.format('neutral'),score=inPercent*10)
      else:
        return render_template('index_v2.html',pred='{}'.format('positive'),score=inPercent*10)
    else:
      inPercent = (1-probability[0])*5+1
      output = round(inPercent)
      print(output,inPercent)
      return render_template('index_v2.html',pred='{}'.format('negative'),score=inPercent*10)

app.run()