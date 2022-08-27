import pandas as pd
import numpy as np

import joblib

df = pd.read_csv('data/credit_risk_prediction_test.csv')
model = joblib.load('src/deployment/models/model_pipeline.pkl')

print(model)

predictions=model.predict(df)


if predictions == [0]:
 print('Credito liberado')
else:
  print('Credito não liberado')