# %% [markdown]
# # Credit-Score-Classification
# 
# - Projeto de classificação de clientes de acordo com seus dados pessoais e financeiros. Dataset disponível em https://www.kaggle.com/laotse/credit-risk-dataset.
# - GitHub : https://github.com/Ewertonv90/Credit-Score-Classification
# 
# 
# 
# # EN
# 
# Problem Statement
# You are working as a data scientist in a global finance company. Over the years, the company has collected basic bank details and gathered a lot of credit-related information. The management wants to build an intelligent system to segregate the people into credit score brackets to reduce the manual efforts.
# 
# Task
# Given a person’s credit-related information, build a machine learning model that classifies the customer if credit can be released or not.
# 
# # PT-BR
# 
# Declaração do problema
# Você está trabalhando como cientista de dados em uma empresa financeira global. Ao longo dos anos, a empresa coletou dados bancários básicos e reuniu muitas informações relacionadas a crédito. A gerência quer construir um sistema inteligente para segregar as pessoas em faixas de pontuação de crédito para reduzir os esforços manuais.
# 
# Tarefa
# Dadas as informações relacionadas ao crédito de uma pessoa, construa um modelo de aprendizado de máquina que possa classificar o cliente e se o crédito deve ser liberado ou não.
# 
# Bussiness Sucess Criteria : more or equal to 85%
#  
# 
# # Data dictonary
# 
#  
# 
# - person_age             =     Age
# - person_income	          =  Annual Income
# - personhomeownership	   =     Home ownership
# - personemplength	        =    Employment length (in years)
# - loan_intent	             =   Loan intent
# - loan_grade	           =     Loan grade
# - loan_amnt	              =  Loan amount
# - loanintrate	          =      Interest rate
# - loan_status	          =      Loan status (0 is non default 1 is default)
# - loanpercentincome	     =   Percent income
# - cbpersondefaultonfile	  =  Historical default
# - cbpresoncredhistlength	=    Credit history length

import pandas as pd
import numpy as np

from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score,accuracy_score, precision_score, recall_score
import joblib
# Data Engeneering

df = pd.read_csv('C:/Users/ewert/Desktop/Credit-Score-Classification/data/credit_risk_dataset.csv')


df = df.drop(columns=['loan_intent','person_emp_length', 'loan_int_rate', 'loan_grade'])
df.head()

df.loc[df['person_age'] >= 100]

index = df[ df['person_age'] >= 100  ].index
df.drop(index, inplace=True)

df = df.dropna(how="all")

df = df.drop_duplicates(keep='first')

df = pd.get_dummies(df, columns=['person_home_ownership', 'cb_person_default_on_file'])


X = df.drop(columns=['loan_status'])
y = df['loan_status']

# Oversampling com tecnica ADASYN

ada = ADASYN(sampling_strategy='auto', random_state=42)
X_res, y_res = ada.fit_resample(X, y)

# Standardazation


scaler = StandardScaler()

df = scaler.fit_transform(df)

df = pd.DataFrame(df)


# Train and Test split


X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.3, random_state=42)



# # Gradient Boosting classifier


clf = GradientBoostingClassifier(criterion='squared_error', learning_rate=0.1, loss='exponential', max_depth=5, min_samples_leaf=5, min_samples_split=10, n_estimators=300, subsample=0.8)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test) 

f1 = round(f1_score(y_test, y_pred, average="micro")*100, 2)
accuracy = round(accuracy_score(y_test, y_pred)*100, 2)
precision = round(precision_score(y_test, y_pred)*100 , 2)
recall = round(recall_score(y_test, y_pred, average='micro')*100 , 2)

print(f"F1 Score: {f1}%")
print(f"Accuracy Score: {accuracy}%")
print(f"Precision Score: {precision}%")
print(f"Recall Score: {recall}%")

metrics = {}

metrics = {

       "accuracy" : accuracy,
       "precision" : precision,
               "recall" : recall,
       "f1" : f1
 }
best_estimator = {}
best_estimator = {'criterion': 'squared_error', 'learning_rate': 0.1, 'loss': 'exponential', 'max_depth': 5, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 300, 'subsample': 0.8}

joblib.dump(best_estimator,filename="src/deployment/models/model_pipeline.pkl",)