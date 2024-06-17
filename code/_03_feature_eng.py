#This script 1) removes extraneous features, 2) scales numerical features, 3) conducts rare label 
  #and 4) one-hot encoding, and 5) finalizes the dataframe


# Load Libraries, Set Options, and Change WD========================================================
## Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import statsmodels.api as sm
import pylab
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


## Options 
pd.options.display.max_columns = 20
pd.options.display.max_rows = 20
pd.options.display.max_colwidth = 80
np.set_printoptions(precision=4, suppress=True)


## Change wd
from pathlib import Path
import os
Path.cwd() #return wd; PosixPath('/Users/keithpost')
root = '/Users/keithpost/Documents/Python/Python projects/spaceship_titanic_ml_py/'
os.chdir(root + 'code') #change wd
Path.cwd() #returns new wd



# Source Functions and Read in Data=================================================================
## Functions
from _00_helper_fns import group_categories


## Data
os.chdir(root + 'data') 
file = open('train_impute.pkl', 'rb')
df0 = pickle.load(file) 


## Check data
df0.describe()
df0.info()



# Feature Selection=================================================================================
## Extraneous variables
objs = ['passenger_group', 'cabin', 'name', 'f_name', 'l_name', 'num', 'name_present']

df = df0.drop(objs, axis=1)
df.info()


## Multicollinearity
nums = ['age', 'room_service', 'food_court', 'shopping_mall', 'spa', 'vr_deck', 'group_size',
'cabin_size']

df_corr = df[nums].corr()
df_corr_signif = df_corr.map(lambda x: '{:.3g}'.format(x)).reset_index().rename(columns={'index': 'variable'})

df_corr_filt = df_corr[(df_corr > 0.9) & (df_corr < 1)]
df_corr_filt = df_corr[df_corr > 0.9]
df_corr_filt #no values--no highly correlated values (only diagonal values are 1)



# Feature Scaling===================================================================================
## Make qqplots of numerical variables: 
### Create grid of subplots
fig, axes = plt.subplots(2, 4, figsize=(8, 6))


### Plot qqplots for each numerical predictor
for i, column_name in enumerate(nums):
    row_index = i // 4
    col_index = i % 4
    ax = axes[row_index, col_index] 
    sm.qqplot(df[column_name], line='s', ax=ax)
    ax.set_title(f'{column_name}')

fig_qqplots = fig

### Adjust layout and display plot
plt.tight_layout()
plt.show(fig_qqplots)
plt.close()
#clearly we see issues with normality, and there are different scales, so this sets up well
    #for normalization (min-max scaling)


## Apply normalization
X_train = df[nums].to_numpy()
scaler= MinMaxScaler().fit(X_train)
scaler
X_scaled = scaler.transform(X_train)

X_scaled.min(axis=0) #min = 0
X_scaled.max(axis=0) #max = 1

df_nums = pd.DataFrame(X_scaled, columns=nums)


# Rare Label Encoding===============================================================================
## Create list of column names of categorical variables
cats = ['ticket', 'home_planet', 'deck', 'side', 'destination', 'floor_4']
cats_id = cats.copy()
cats_id.insert(0, 'passenger_id')


## Get counts of data
df_cats = df[cats]
df_cats_long = pd.melt(df[cats_id], id_vars="passenger_id")
df_cats_n = df_cats_long.drop('passenger_id', axis=1).groupby(['variable', 'value'],
                                                              as_index=False).size()
df_cats_n = df_cats_n.rename(columns={'value': 'category', 'size': 'count'})


## Make barplots
### Create a list of colors
colors = ['blue', 'green', 'red', 'purple', 'orange', 'black']

### Create a subplot grid
fig, axes = plt.subplots(2, 3, figsize=(8.5, 7))

### Build each subplot using a for loop
for i, variable in enumerate(cats):
  #create each df and sort them in descending count order
  df_plot = df_cats_n[df_cats_n['variable']==variable]
  df_plot = df_plot.sort_values(by='count', ascending=False)
  
  #create row and col indexes
  row = i // 3
  col = i % 3
  
  #select a color
  color=colors[i]
  
  #build the barplot
  axes[row, col].bar(df_plot['category'], df_plot['count'], color=color)
  
  #add labels
  axes[row, col].set_xlabel('Category')
  axes[row, col].set_ylabel('Count')
  axes[row, col].set_title(f'{variable}')
  
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 2].tick_params(axis='x', rotation=45)

### Adjust spacing between subplots
plt.subplots_adjust(wspace=0.05, hspace=0.05)  

fig_rle = fig

plt.tight_layout()
plt.show(fig_rle)
plt.close() 
#this shows that ticket and deck have rare categories (< 2%; 174)

## Make tables of frequencies
df_cats_n[df_cats_n['variable']==cats[0]] #ticket: 5-8 are below threshold
df_cats_n[df_cats_n['variable']==cats[2]] #deck: T is below threshold

8693*.02 #174 = threshold


## Group rare categories
#create lists of rare categories
rare_tickets = ['05', '06', '07', '08']
rare_decks = ['A', 'T']

#group them
df['ticket'] = df['ticket'].apply(group_categories, rare_cats=rare_tickets, new_cat='05_08')
df['ticket'] = pd.Categorical(df['ticket'], categories=['01', '02', '03', '04', '05_08'])
df['ticket'].value_counts()

df['deck'] =  df['deck'].apply(group_categories, rare_cats=rare_decks, new_cat='A_T')
df['deck'] = pd.Categorical(df['deck'], categories=['B', 'C', 'D', 'E', 'F', 'G', 'A_T'])
df['deck'].value_counts()



# Label Encoding====================================================================================
## Preview categorical data
df_cats = df[cats]
df_cats.apply(pd.Series.unique)
#although a few of these features have a number/letter component to them (e.g., ticket, deck,
  #floor_4), I don't see any inherent ranks so will opt for one-hot encoding
  
  
## One-hot encoding
encoder = OneHotEncoder() #create instance of 'OneHotEncoder'
# encoder.fit(df_cats) #fit encoder to categorical data
ar_cats_encoded = encoder.fit_transform(df_cats).toarray() #fit and transform data
df_cats_encoded = pd.DataFrame(ar_cats_encoded, 
                               columns=encoder.get_feature_names_out(cats)) #convert to DF
df_cats_encoded
                               
                               
# Finalize Feature Engineering======================================================================
## Boolean vars
#create list of boolean variables
bools_tv = ['cryo_sleep', 'vip']

#convert to floats
df_bools = df[bools_tv].astype('float64')


## ID var
df_id = df[['passenger_id']]


## Target var
df_tv = df[['transported']]


## Combine all 
df_final = pd.concat([df_id, df_nums, df_cats_encoded, df_bools, df_tv], axis=1)
df_final.info()


## Turn column names into DF for report
final_col_nm = df_final.columns[0:36]
ar_final_col_nm = np.array(final_col_nm).reshape(9, 4)
df_reshaped_col_nm = pd.DataFrame(ar_final_col_nm, columns=["", "", "", ""])
df_reshaped_col_nm


# Save Data to File=================================================================================
#save in pickle format to retain data types and categories
# afile = open('train_final.pkl', 'wb')
# pickle.dump(df_final, afile)
# afile.close()










