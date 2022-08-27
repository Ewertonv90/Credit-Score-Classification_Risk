from tkinter.ttk import Separator
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def risk_credit_predict():
    dados = request.get_json()
    data = dados["data"]
   
    df = pd.DataFrame(data, index=[0])

    model = joblib.load('src/deployment/models/model_pipeline.pkl')
    predictions=model.predict(df)

    output = predictions[0]

    if predictions == [0]:
     return { "message" : "Parabéns, você não oferece risco de inadimplemento!", "prediction" : 0}, 200
    
    if predictions == [1]:
     return { "message" : "Ops!, infelizmente não foi dessa, vez, tente novamente em 6 meses", "prediction" : 1}, 200
    else:
     return { "message" : "Erro: Dados preeenchidos incorretamente, tente novamente", "prediction" : "error "}, 401

if __name__ == '__main__':
 app.run(debug=True)