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

# %% [markdown]
# # 1.1 Data Undestanding

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px



# %%
df = pd.read_csv('C:/Users/ewert/Desktop/Credit-Score-Classification/data/credit_risk_dataset.csv')


# %%
df.describe()


# %% [markdown]
# - Média salaria de $ 6.607 dolares anuais
# - Esses salarios variam 6.1983 dolares anuais para mais ou para menos.
# - Como se é esperado, quanto maior a idade, maior o salário.

# %%
df.info()


# %%
df.shape

# %% [markdown]
# # EDA 

# %%
df.isnull().sum()

# %%
print('linhas duplicadas:',len(df)-len(df.drop_duplicates()))

# %%
np.unique(df['loan_status'], return_counts=True)

sns.countplot(x = df['loan_status'])


# %% [markdown]
# - O target Loan Status é o nosso alvo pra determinar o risco de crédito, caso seja 0 o valor, o emprestimo está em dia, caso seja 1, o cliente não pagou a dívida. Préviamente, 25473 clientes da base de dados pagaram suas dívidas, e 7108 estão em atraso. Isso representa cerca de 78,18% de pessoas em dia e 21,82% possuem dívidas em atraso, somando o total de 32581 registros de clientes. Há um desbalanceamento entre as duas classes, necessitando de técnica de oversampling.

# %%
plt.hist(x=df['person_age'])

# %% [markdown]
# - A maioria dos clientes tem entre 20 e 42 anos

# %%
df.hist(bins= 15, figsize= (20,20))

# %%
correlacao = df.corr()

# %%
plt.figure(figsize=(7,7))
sns.heatmap(correlacao, annot=True)

# %%
sns.countplot(x = df['person_home_ownership'])
df['person_home_ownership'].groupby(df['person_home_ownership']) .count()

# %% [markdown]
# - 41,26% dos clientes moram em imóveis hipotecados.
# - 50,48% dos  clientes moram em imóveis alugados.
# - 7,94% dos clientes tem casa própria quitada.
# - 0,32% dos clientes tem imóveis em situações diversas, diferentes das principais citadas acima.

# %%
graphic = px.scatter_matrix(df, dimensions=['person_age','person_income', 'loan_amnt'], color= 'loan_status')
graphic.show()

# %%
graphic = px.scatter_matrix(df, dimensions=['loan_amnt','person_emp_length'], color= 'loan_status')
graphic.show()

# %%
plt.boxplot(df['person_age'])


# %% [markdown]
# - Já identificada presença de outiliers, como pessoas de 123 e 144 anos. Além disso, a despadronização dos dados pode fazer com oque o algoritmo pense que uma feature tenha mais importancia que a outra, tendo necessidade de utlizar a padronização de dados. Devido a presença apenas de numeros naturais, a padronização escolhida será o Standardzation.

# %% [markdown]
# # Data preparation

# %% [markdown]
# Preparação do dataset para modelagem
# 
# - Retirada de campos desnecessários para atingir o alvo: loan_grade e loan_intent.
# - retirada de dados nulos.
# - descarte de linhas duplicadas ( 165 linhas )
# - Retirada de outliers na coluna Age: idades entre 100 e  144 anos de idade, retirada de idades negativas.
# - Dummificação de dados categóricos(One hot encoding): person_home_ownership e cb_person_default_on_file
# - alteração dos tipos de dados object para float 
# - Padronização dos dados com standard Scale.
# - Balanceamento da variável target "loan_status" com a técnica de oversampling ADASYN.

# %%
df = df.drop(columns=['loan_intent','person_emp_length', 'loan_int_rate', 'loan_grade'])
df.head()

# %%
df.loc[df['person_age'] >= 100]

# %%
index = df[ df['person_age'] >= 100  ].index
df.drop(index, inplace=True)

# %%
df = df.dropna(how="all")

# %%

total_nan_values = df.isnull().sum()
print (total_nan_values)

# %%
df.loc[df['person_age'] >= 100]

# %%
df = df.drop_duplicates(keep='first')

# %%
print('linhas duplicadas:',len(df)-len(df.drop_duplicates()))

# %%
df.head()

# %%
df = pd.get_dummies(df, columns=['person_home_ownership', 'cb_person_default_on_file'])
# cb_person_default_on_file

df

# %%
df.dtypes

# %%
X = df.drop(columns=['loan_status'])
y = df['loan_status']
X.dropna()
y.dropna()

# %% [markdown]
# ## Balanceamento dos dados com ADASYN

# %%
from imblearn.over_sampling import ADASYN

shape_original = y.shape

print(f'Shape dataset original {shape_original} linhas')

ada = ADASYN(sampling_strategy='auto', random_state=42)
X_res, y_res = ada.fit_resample(X, y)

shape_resample = y_res.shape
print(f'Shape dataset balanceado {shape_resample} linhas')

# %% [markdown]
# ## Padronização dos dados com Standard Scale

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df = scaler.fit_transform(df)

df = pd.DataFrame(df)



# %%
df.head()

# %%
total_nan_values = df.isnull().sum()
print (total_nan_values)

# %% [markdown]
# # Data modeling
# 
# Modelagem de dados será realizada utilizando as melhores técnicas de classificação do mercado:
# 
# - Regressão logística
# - Random forest
# - Gradient Boosting
# - Extreme Gradient Boosting(XGBoost)
# 
# ### Metricas de avaliação dos modelos:
# 
# - F1 Score
# - Accuracy Score
# - Precision Score
# - Recall Score

# %%


# %% [markdown]
# # Separando 33% dos dados para treinamento

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.3, random_state=42)

print('X_train: ', X_train.shape)
print('X_test: ', X_test.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)

# %% [markdown]
# # Regressão logística

# %%
from sklearn.model_selection import GridSearchCV

params = {
    'C': (0.01, 1, 10, 100),
    'penalty': ('l1','l2', None)
}

# %%
X.isnull().values.any() and y.isnull().values.any()

# %%
from sklearn.linear_model import LogisticRegression

lrc = LogisticRegression()

grid_search = GridSearchCV(estimator=lrc, param_grid= params , n_jobs= -1 , cv=3 , verbose= 0 )

grid_search.fit(X_train, y_train)

# %%
print(grid_search.best_params_)
best_rf = grid_search.best_estimator_

# %% [markdown]
# # Métricas do modelo treinado com Regressão logística

# %%
from sklearn.metrics import f1_score,accuracy_score, precision_score, recall_score

y_pred = best_rf.predict(X_test)

f1 = round(f1_score(y_test,y_pred, average='micro')*100, 2)
accuracy = round(accuracy_score(y_test, y_pred)*100, 2)
precision = round(precision_score(y_test, y_pred)*100,2)
recall = round(recall_score(y_test, y_pred, average='micro')*100,2)

print(f"F1 Score: {f1}%")
print(f"Accuracy Score: {accuracy}%")
print(f"Precision Score: {precision}%")
print(f"Recall Score: {recall}%")

metrics = {}

metrics['LogisticRegression'] = {
    'f1' : f1,
    'accuracy' : accuracy,
    'precision' : precision,
    'recall' : recall
}

# %%
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(best_rf, X_test, y_test, values_format='d')  
plt.grid(False)
plt.show()

# %% [markdown]
# # Random Forest classifier

# %%
from sklearn.ensemble import RandomForestClassifier

params = {
    'max_depth' : [100, 150],
    'max_features' : [3, 4, 5],
    'criterion' : ['gini', 'entropy'],
    'min_samples_leaf' : [1, 2, 3],
    'min_samples_split' : [2, 3, 8, 10],
    'n_estimators' : [200, 300, 400]
}

# %%
from skopt import BayesSearchCV

rfc = RandomForestClassifier(criterion='entropy', max_depth=150, max_features=5, min_samples_leaf=3, min_samples_split=10, n_estimators=400 )

rfc.fit(X_train, y_train)

#bayes_search = BayesSearchCV(clf, search_spaces= params, n_jobs=-1, cv=3, verbose=0 )

#bayes_search.fit(X_train, y_train)
#('criterion', 'entropy'), ('max_depth', 150), ('max_features', 5), ('min_samples_leaf', 3), ('min_samples_split', 10), ('n_estimators', 400)

# %%
#print(bayes_search.best_params_)

#best_rf = bayes_search.best_estimator_

# %% [markdown]
# # Metricas do modelo treinado com Random Forest Classifier

# %%
y_pred = rfc.predict(X_test)

f1 = round(f1_score(y_test, y_pred, average="micro")*100, 2)
accuracy = round(accuracy_score(y_test, y_pred)*100, 2)
precision = round(precision_score(y_test, y_pred)*100 , 2)
recall = round(recall_score(y_test, y_pred, average='micro')*100 , 2)

print(f"F1 Score: {f1}%")
print(f"Accuracy Score: {accuracy}%")
print(f"Precision Score: {precision}%")
print(f"Recall Score: {recall}%")

metrics["RandomForest"] = {

       "accuracy" : accuracy,
       "precision" : precision,
       "recall" : recall,
       "f1" : f1
 }

# %%
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(rfc, X_test, y_test,  values_format='d')  
plt.grid(False)
plt.show()

# %% [markdown]
# # Gradient Boosting classifier

# %%
from sklearn.ensemble import GradientBoostingClassifier

params = {
    'n_estimators' : [100, 200, 300],
    'learning_rate' : [0.1, 0.5, 1.0],
    'subsample': [0.1,  0.3, 0.5, 0.8, 1.0],
    'loss': ['log_loss', 'deviance', 'exponential'],
    'criterion' : ['squared_error', 'mse'],
    'max_depth' : [1, 3, 5, 10],
    'min_samples_leaf' : [1,3, 5, 10],
    'min_samples_split': [1, 3, 5, 10]
}

# %%
from sklearn.model_selection import GridSearchCV

clf = GradientBoostingClassifier(criterion='mse', learning_rate=0.1, loss='exponential', max_depth=5, min_samples_leaf=5, min_samples_split=10, n_estimators=300, subsample=0.8)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test) 

# %%
#grid_search = GridSearchCV(estimator=clf, param_grid= params , n_jobs= -1 , cv=3 , verbose= 0 )

#grid_search.fit(X_train, y_train)

# %%
#print(grid_search.best_params_)
#best_rf = grid_search.best_estimator_
#{'criterion': 'mse', 'learning_rate': 0.1, 'loss': 'exponential', 'max_depth': 5, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 300, 'subsample': 0.8}

# %% [markdown]
# # Metricas do modelo treinado com Gradient boosting Classifier

# %%
#y_pred = best_rf.predict(X_test)

f1 = round(f1_score(y_test, y_pred, average="micro")*100, 2)
accuracy = round(accuracy_score(y_test, y_pred)*100, 2)
precision = round(precision_score(y_test, y_pred)*100 , 2)
recall = round(recall_score(y_test, y_pred, average='micro')*100 , 2)

print(f"F1 Score: {f1}%")
print(f"Accuracy Score: {accuracy}%")
print(f"Precision Score: {precision}%")
print(f"Recall Score: {recall}%")

metrics["Gradient_boosting"] = {

       "accuracy" : accuracy,
       "precision" : precision,
               "recall" : recall,
       "f1" : f1
 }

# %%
plot_confusion_matrix(clf, X_test, y_test, values_format='d')  
plt.grid(False)
plt.show()

# %% [markdown]
# # Extreme Gradient Boosting (XGBoost)

# %%
%pip install xgboost

# %%
from xgboost import XGBClassifier

xgb = XGBClassifier(booster='gbtree', gamma=0, learning_rate=0.1, max_depth=6, use_rmm=True, validate_parameters=False, verbosity=0)
xgb.fit(X_train, y_train)

#{'booster': 'gbtree', 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 6, 'use_rmm': True, 'validate_parameters': False, 'verbosity': 0}
params = {
   'booster': ['gbtree', 'gblinear', 'dart'],
   'verbosity' : [0],
   'learning_rate': [0.1, 0.5, 1.0],
   'max_depth': [1, 3, 6, 8, 10],
   'use_rmm': [True, False],
   'validate_parameters': [False, True],
   'gamma': [0, 10, 50, 100, 300, 500],
   

}


# %%
#grid_search = GridSearchCV(estimator=xgb, param_grid= params , n_jobs= -1 , cv=3 , verbose= 0 )

#grid_search.fit(X_train, y_train)

# %%
#print(grid_search.best_params_)
#best_rf = grid_search.best_estimator_
#{'booster': 'gbtree', 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 6, 'use_rmm': True, 'validate_parameters': False, 'verbosity': 0}

# %% [markdown]
# # Metricas do modelo treinado com XGBoost Classifier

# %%
#y_pred = best_rf.predict(X_test)
y_pred = xgb.predict(X_test)
f1 = round(f1_score(y_test, y_pred, average="macro")*100, 2)
accuracy = round(accuracy_score(y_test, y_pred)*100, 2)
precision = round(precision_score(y_test, y_pred)*100 , 2)
recall = round(recall_score(y_test, y_pred, average='micro')*100 , 2)

print(f"F1 Score: {f1}%")
print(f"Accuracy Score: {accuracy}%")
print(f"Precision Score: {precision}%")
print(f"Recall Score: {recall}%")

metrics["XGBoost"] = {

       "accuracy" : accuracy,
       "precision" : precision,
       "recall" : recall,
       "f1" : f1
 }

# %%
plot_confusion_matrix(xgb, X_test, y_test, values_format='d')  
plt.grid(False)
plt.show()

# %%
metrics

# %% [markdown]
# # Data Evaluation
# - Assessment of data
# - Mining results
# - Bussiness success criteria is done?
# - Review process
# - Next steps
# - List of possible actions
# - Decision
# 
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

# %% [markdown]
# # Assessment of data

# %%
metrics_df = pd.DataFrame(metrics)
metrics_df.head()

# %%
print('Media das métricas por tipo:')
print('Logistic Regression: ',round(metrics_df['LogisticRegression'].mean(),2)) 
print('Random Forest: ',round(metrics_df['RandomForest'].mean(),2))
print('Gradient Boosting: ',round(metrics_df['Gradient_boosting'].mean(),2))
print('XGBoost: ',round(metrics_df['XGBoost'].mean(),2))


# %% [markdown]
# # Mining Results
# 
# 
# - Maior Accuracy Score :  Gradient Boosting                 88,23%
# - Maior Precision Score : XGBoost                           91,18%
# - Maior Recall Score :    Gradient Boosting                 88,23%
# - Maior F1 Score:         Gradient Boosting                 88,23%
# 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# 
# - Menor Accuracy Score :  Regressão Logística                                 80,74% 
# - Menor Precision Score : Regressão Logística                                 73,19% 
# - Menor Recall Score :    Regressão Logística                                 80,74%
# - Menor F1 Score:         XGBoost   ----------------------------------------  78,24%
# 
# # Metric accuracy average
# 
# - Regressão Logística :  78,85%
# - Random Forest:         88,38%
# - Gradient Boosting:     88,45%
# - XGBoost :              86,33%
# 
# # Bussiness success criteria and approved models
# - XGBoost :              86,33%
# - Random Forest:         88,38%
# - Gradient Boosting:     88,45%
# 
# # model chosen for the product
# 
# - Gradient Boosting:  88,45%

# %% [markdown]
# # Next steps
# 
# - Ajustes finos nos hiperparâmetros e mais opções no Cross Validation, aumentando a taxa de sucesso.
# - aumentar a base de treino e melhorar acuracia de verdadeiros positivos e verdadeiros negativos.
# - inserir novos hiperparametros e melhorar a acuracia dos resultados

# %% [markdown]
# #


