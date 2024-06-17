# This script processes the test data by going through the same steps as the training data: 
  #1) performs initial wrangling, 2) discretizes variable num, 3) creates 'size' features, 
  #4) imputes data, 5) removes extraneous features, 6) scales numerical features, performs 
  #7) rare-label and 8) one-hot encoding, and 9) finalizes the dataframe
  

# Load Libraries, Set Options, & Source Fns=========================================================
## Load libraries
import pandas as pd
import inflection
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


## Options 
pd.options.display.max_columns = 20
pd.options.display.max_rows = 20
pd.options.display.max_colwidth = 80
np.set_printoptions(precision=4, suppress=True)


## Functions
from pathlib import Path
import os
Path.cwd() #return wd; PosixPath('/Users/keithpost')
root = '/Users/keithpost/Documents/Python/Python projects/spaceship_titanic_ml_py/'
os.chdir(root + 'code') 
Path.cwd()
from _00_helper_fns import group_categories


# Import Data=======================================================================================
## Change wd
os.chdir(root + 'data') 


## Import data
df0 = pd.read_csv("test.csv") 
#convert to snake_case
df0.columns = df0.columns.map(inflection.underscore) 


## Check data
df0.describe()
df0.info()



# Initial Wrangling=================================================================================
## Copy DF
df1 = df0.copy()


## Split objects into multiple columns
df1[['passenger_group', 'ticket']] = df1.passenger_id.str.split('_', n=1, expand=True)
df1[['deck', 'num', 'side']] = df1.cabin.str.split('/', n=2, expand=True)
df1[['f_name', 'l_name']] = df1.name.str.split(' ', n=1, expand=True) 



# Discretize num====================================================================================
## Cut/bin num by various sizes
### Extract floor information
df = df1.copy()
df['num'] = df['num'].astype('float64')
df['floor'] = df['num']//100 #quotient from integer division
floors = df['floor'].unique().tolist()
floors #0-18 + nans


### Create bins for 4-floor groups
#### Create integer lists for binning
bin_floors_4 = np.arange(0, 21, 4).tolist()
bin_floors_4


#### Create bin labels
#set prefix
prefix = "floors_"

#create bin labels using list comprehension
floor_4_lab = [f"{prefix}{i}_{i+4}" for i in bin_floors_4[:-1]]


### Create new variable and drop 'floor' 
df['floor_4'] = pd.cut(df['floor'], bins=bin_floors_4, labels=floor_4_lab, right=False)

#now the categories are listed using labels
df['floor_4']

#drop 'floor'
df_f = df.drop(['floor'], axis=1)



# Create New 'Size' Variables=======================================================================
## Check for NAs-------------------------
df_f[['passenger_group', 'cabin']].isna().sum() 
#0 for pg and 100 for cabin


## Group size-------------------------
df_pg_gs = df_f[['passenger_group']].groupby('passenger_group', as_index=False).agg(
  group_size=('passenger_group', 'size'))


## Cabin size-------------------------
df_cab_cs = df_f[df_f['cabin'].notna()][['cabin']].groupby('cabin', as_index=False).agg(cabin_size=('cabin', 'size'))
df_cab_cs.sum() #4177 (excludes missing values)


## Join them in
df_f_gs = pd.merge(df_f, df_pg_gs, on='passenger_group')
df_fs = pd.merge(df_f_gs, df_cab_cs, on='cabin', how='left')
#s = size



# Data Imputation (numerical and categorical)=======================================================
## Prepare data for imputation
### Remove extraneous columns
cols_to_drop = ['passenger_id', 'passenger_group', 'cabin',  'name', 'f_name', 'l_name', 'num']

df_fs_imp = df_fs.drop(cols_to_drop, axis=1)


## Convert cols to appropriate types
### Boolean
# df_fs_imp[['cryo_sleep', 'vip']] = df_fs_imp[['cryo_sleep', 'vip']].astype('category')


### Categorical
cats = ['ticket', 'home_planet', 'deck', 'side', 'destination', 'floor_4']
df_fs_imp[cats] = df_fs_imp[cats].astype('category')


### Numerical
nums = df_fs_imp.columns.drop(cats).tolist()
df_fs_imp[nums] = df_fs_imp[nums].astype('float64')


## Split data
df_imp_nums = df_fs_imp[nums]
df_imp_cats = df_fs_imp[cats]


## Impute data using median for numerical columns and most frequent for categorical ones
#numerical
num_imputer = SimpleImputer(missing_values=np.nan, strategy="median")
df_imp_nums[:] = num_imputer.fit_transform(df_imp_nums)
df_imp_nums.info() #all values are imputed

#categorical
cat_imputer = SimpleImputer(strategy="most_frequent")
df_imp_cats[:] = cat_imputer.fit_transform(df_imp_cats)
df_imp_cats.info()

#combine DFs
df_fs_imp = pd.concat([df_fs[cols_to_drop], df_imp_nums, df_imp_cats], 
                       axis="columns").reset_index(drop=True)

#convert cryo_sleep & vip from category back to Boolean
df_fs_imp[['cryo_sleep', 'vip']] = df_fs_imp[['cryo_sleep', 'vip']].astype('bool')


## Check for missingness
df_fs_imp.info()
#columns with missing values: cabin, name, f_name, l_name, num



# Feature Selection=================================================================================
#drop extraneous variables
del df

objs = ['passenger_group', 'cabin', 'name', 'f_name', 'l_name', 'num']

df = df_fs_imp.drop(objs, axis=1)
df.info() #no NA values



# Feature Scaling===================================================================================
#apply normalization (min-max scaling)
del nums
nums = ['age', 'room_service', 'food_court', 'shopping_mall', 'spa', 'vr_deck', 'group_size',
'cabin_size']

X_test = df[nums].to_numpy()
scaler= MinMaxScaler().fit(X_test)
scaler
X_scaled = scaler.transform(X_test)

df_nums = pd.DataFrame(X_scaled, columns=nums)



# Rare Label Encoding===============================================================================
#group rare categories of tickets and decks

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
#one-hot encoding
df_cats = df[cats] #isolate categorical variables

encoder = OneHotEncoder() 
ar_cats_encoded = encoder.fit_transform(df_cats).toarray() 
df_cats_encoded = pd.DataFrame(ar_cats_encoded, 
                               columns=encoder.get_feature_names_out(cats)) 
df_cats_encoded



# Finalize Feature Engineering======================================================================
## Boolean vars
#create list of boolean variables
bools_tv = ['cryo_sleep', 'vip']

#convert to floats
df_bools = df[bools_tv].astype('float64')


## ID var
df_id = df[['passenger_id']]


## Combine all 
df_final = pd.concat([df_id, df_nums, df_cats_encoded, df_bools], axis=1)
df_final.info()



# Save Data to File=================================================================================
#save in pickle format to retain data types and categories
# afile = open('test_prepped.pkl', 'wb')
# pickle.dump(df_final, afile)
# afile.close()














