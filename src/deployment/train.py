from gettext import install
import pip

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('C:/Users/ewert/Desktop/Credit-Score-Classification/raw dataset/credit_risk_dataset.csv')

df = df.drop(columns=[ 'loan_grade'])

df = df.drop(columns=['loan_intent','person_emp_length', 'loan_int_rate'])

df = df.drop(81)
df = df.drop(183)
df = df.drop(575)
df = df.drop(747)
df = df.drop(32297)

df = pd.get_dummies(df, columns=['person_home_ownership', 'cb_person_default_on_file'])

X = df.drop(columns=['loan_status'])
X.dropna()


clf = RandomForestClassifier(criterion='entropy', max_depth= 150, max_features=3, min_samples_leaf= 1, min_samples_split=10, n_estimators=300)

clf.predict(X)


df_final = pd.DataFrame(X)

df_final.to_excel()

