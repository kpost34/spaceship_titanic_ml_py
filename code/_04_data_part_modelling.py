#This script 1) assesses three model types, 2) tunes two of them, 3) selects the best model, and
  #4) performs model diagnostics


# Load Libraries, Set Options, and Change WD========================================================
## Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


## Options 
pd.options.display.max_columns = 20
pd.options.display.max_rows = 20
pd.options.display.max_colwidth = 80
np.set_printoptions(precision=4, suppress=True)



# Read in Data======================================================================================
## Change wd
from pathlib import Path
import os
Path.cwd() #return wd; PosixPath('/Users/keithpost')
root = '/Users/keithpost/Documents/Python/Python projects/spaceship_titanic_ml_py/'


## Data
os.chdir(root + 'data') 
file = open('train_final.pkl', 'rb')
df0 = pickle.load(file) 


## Check data
df0.describe()
df0.info()



# Modelling=========================================================================================
#set random number
random.seed(5)

## Split up predictors and target variable
y = pd.Series(df0['transported']).array
X = df0.drop(['passenger_id', 'transported'], axis=1).to_numpy()

## Logistic regression-------------------------
### Evaluate using entire training set
mod_logreg = LogisticRegression() #create model
scores_logreg = cross_val_score(mod_logreg, X, y, cv=5)
scores_logreg.mean() #.7862


## Decision tree-------------------------
mod_dt = DecisionTreeClassifier()
scores_dt = cross_val_score(mod_dt, X, y, cv=5)
scores_dt.mean() #.7223


## Random forest-------------------------
mod_rf = RandomForestClassifier()
scores_rf = cross_val_score(mod_rf, X, y, cv=5)
scores_rf.mean() #.7913

#choose logistic regression and random forest models for tuning



# Hyperparameter tuning=============================================================================
## Logistic regression model-------------------------
## Create parameter grid
param_grid_logreg = {'penalty': ['l2', 'l1'], #2
                     'max_iter': [100, 500, 1000], #3
                     'C': [0.01, 0.1, 1, 10, 100], #5
                     'solver': ['liblinear', 'saga']} #2
grid_logreg = GridSearchCV(LogisticRegression(), param_grid_logreg, cv=5)
#penalty: L2 = ridge regularization; L1 = lasso regularization
#max_iter: max number of iterations to optimize model parameters
#C: inverse of regularization strength; smaller C = stronger regularization
#solver: approach for convergence


## Fit model to cross-validated folds
grid_logreg.fit(X, y)


## Results and best model
cv_results_logreg = grid_logreg.cv_results_ 

df_results_logreg = pd.DataFrame({
  'hyperparameters': cv_results_logreg['params'],
  'mean_accuracy': cv_results_logreg['mean_test_score']
})

# View(df_results_logreg)

grid_logreg.best_score_ #.7918
grid_logreg.best_params_
# {'C': 0.1, 'max_iter': 500, 'penalty': 'l1', 'solver': 'saga'}
#moderate C, within 500 iterations for convergence, ridge regularization, saga is used in L1
  # regularization

#regularization: used to prevent overfitting; adds penalty term to model's loss function to encourage 
  #it to learn simpler patterns and reduces complexity of learned relationships between features
  #and target variable: 
  #L1 regularization: penalty term = sum of abs values of coefficients of model; encourages model
    #sparsity--coefficients close to 0; helpful with high-d datasets or when irrelevant features
    #should be disregarded
  #L2: penalty term = sum of squared values of the model coefficients; penalized large coeffs more
    #than small ones; tends to distribute weights more evenly across all features
  
#saga is well-suited for problems with L1 regularization and can handle both sparse & dense data

#note: liblinear: efficient for relatively small samples and set of features; can specify either L1
  #or L2 regularization, can also handle sparse data and large numbers of features
  

## Random forest model-------------------------
param_grid_rf = {'n_estimators': [50, 100, 500], #3
                 'min_samples_split': [2, 5, 10, 20], #4
                 'min_samples_leaf': [2, 5, 10, 20]} #4
grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5)
#n_estimators: number of trees
#min_samples_split: min number of samples required to split an internal node
#min_samples_leaf: min number of samples required at a leaf node


## Fit model to cross-validated folds
grid_rf.fit(X, y)


## Results and best model
cv_results_rf = grid_rf.cv_results_ 

df_results_rf = pd.DataFrame({
  'hyperparameters': cv_results_rf['params'],
  'mean_accuracy': cv_results_rf['mean_test_score']
})

# View(df_results_rf)

grid_rf.best_score_ #.8004
grid_rf.best_params_
# {'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 100}



# Finalize Model and Perform Diagnostics============================================================
## Select best model
best_model_rf = grid_rf.best_estimator_


## Confusion matrix
### Create confusion matrix
y_pred = best_model_rf.predict(X)
conf_matrix = confusion_matrix(y, y_pred)


### Plot confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
plt.close()


## Feature importance
#grab feature names and importances
feat_names = df0.drop(['passenger_id', 'transported'], axis=1).columns.tolist()
feat_import = best_model_rf.feature_importances_

#sort them
indices = np.argsort(feat_import)[::-1]

#create a DataFrame to store feature importances with feature names
df_feat_import = pd.DataFrame({'Feature': feat_names, 'Importance': feat_import})
df_feat_import_sorted = df_feat_import.sort_values(by='Importance', ascending=False)
df_feat_import_sorted

#plot them
plt.barh(df_feat_import_sorted['Feature'], df_feat_import_sorted['Importance'], color='royalblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.gca().invert_yaxis()  
plt.tight_layout()
plt.show()
plt.close()



# Write Modelling Objects to Files==================================================================
## Change WD
os.chdir(root + 'modelling') 
Path.cwd()


## Save in pickle format to retain data types and categories
afile = open('best_model_rf.pkl', 'wb')
pickle.dump(best_model_rf, afile)
afile.close()







