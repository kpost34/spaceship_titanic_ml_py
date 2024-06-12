#This script 1) predicts target variable (transported) and 2) writes file containing predicted
  #values for assessment


# Load Libraries, Set Options, & Import Data & Model================================================
## Load libraries
import pandas as pd
import numpy as np
import pickle


## Options 
pd.options.display.max_columns = 20
pd.options.display.max_rows = 20
pd.options.display.max_colwidth = 80
np.set_printoptions(precision=4, suppress=True)


## Data
from pathlib import Path
import os
Path.cwd() #return wd; PosixPath('/Users/keithpost')
root = '/Users/keithpost/Documents/Python/Python projects/spaceship_titanic_ml_py/'
os.chdir(root + 'data') 
Path.cwd()
file = open('test_prepped.pkl', 'rb')
df0 = pickle.load(file) 


## Model
os.chdir(root + 'modelling') 
file2 = open('best_model_rf.pkl', 'rb')
best_model_rf = pickle.load(file2) 



# Predict transported===============================================================================
## Prep test data for prediction
df = df0.drop('passenger_id', axis=1)
X = df.to_numpy()


## Predict y values
y_pred = best_model_rf.predict(X)


## Combine ids with predicted values
df_predict = pd.concat([df0[['passenger_id']], pd.DataFrame(y_pred)], axis=1)
df_predict.columns = ['PassengerId', 'Transported']



# Write to File=====================================================================================
os.chdir(root + 'data') 
df_predict.to_csv('predicted_transported.csv', index=False)
#.73299, which indicates evidence of overfitting (because this model had an accuracy score of
  #.8004 on training data)



