# This script 1) discretizes variable num, 2) explores string imputation, 3) creates 'size' 
  #features, and 4) imputes data


# Load Libraries, Set Options, and Change WD========================================================
## Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer
import pickle


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
from _00_helper_fns import make_grouped_barplot, littles_mcar_test


## Data
os.chdir(root + 'data') 
file = open('train_initial_clean.pkl', 'rb')
df0 = pickle.load(file) 



# Discretize num====================================================================================
#context: num is the room number of the cabin (which also contains the deck and side); num ranges
  #from 1-1800+, but these are strings. They could easily be alphabetical or alphanumerical codes. 
  #strings with 1800+ categories have little value in prediction, but they can be discretized into
  #a reasonable number of groups/bins, making it a factor variable. Let's do this coarsely by
  #splitting num into groups by counts of 100, 200, 300, or 400. But, instead of simply dividing
  #by these values, let's consider the hundreds digit a floor, and group by an individual floor
  #up to 4 floors to make the values more interpretable
  
## Look more closely at 'num' field
#group by counts
df_num_n = df0[['num']].value_counts().reset_index()
df_num_n.head(20)
#shows that num values are not distinct and can occur up to 28 times in the training data


## Cut/bin num by various sizes
### Extract floor information
df = df0.copy()
df['num'] = df['num'].astype('float64')
df['floor'] = df['num']//100 #quotient from integer division
floors = df['floor'].unique().tolist()
floors #0-18 + nans
min(floors) #0
max(floors) #18...so 21/22 depending on bin width


### Create bins for 2, 3, and 4-floor groups
#### Create integer lists for binning
bin_floors_2 = np.arange(0, 21, 2).tolist()
bin_floors_3 = np.arange(0, 22, 3).tolist()
bin_floors_4 = np.arange(0, 21, 4).tolist()

bin_floors_2
bin_floors_3
bin_floors_4


#### Create bin labels
#set prefix
prefix = "floors_"

#create bin labels using list comprehension
floor_2_lab = [f"{prefix}{i}_{i+2}" for i in bin_floors_2[:-1]]
floor_3_lab = [f"{prefix}{i}_{i+3}" for i in bin_floors_3[:-1]]
floor_4_lab = [f"{prefix}{i}_{i+4}" for i in bin_floors_4[:-1]]


### Create new variables/columns using said groups
df['floor_2'] = pd.cut(df['floor'], bins=bin_floors_2, labels=floor_2_lab, right=False)
df['floor_3'] = pd.cut(df['floor'], bins=bin_floors_3, labels=floor_3_lab, right=False)
df['floor_4'] = pd.cut(df['floor'], bins=bin_floors_4, labels=floor_4_lab, right=False)

#now the categories are listed using labels
df['floor_2'] 
df['floor_3']
df['floor_4']

#convert floor (a float) into category type
df['floor'] = df['floor'].astype('category')

df[['num',  'floor', 'floor_2', 'floor_3', 'floor_4']].value_counts().tail(20)
#see that binning is working correctly


### Visualize relationships between floor bins and transportation status
#### Calculate counts using floor bins and transported status
#create list object for wrangling
cats_floor = ["floor", "floor_2", "floor_3", "floor_4"]
cats_floor_id_tv = cats_floor.copy()
cats_floor_id_tv.extend(["passenger_id", "transported"])

#pivot to long form
df_floor_long = pd.melt(df[cats_floor_id_tv], id_vars=["passenger_id", "transported"])
df_floor_long.dtypes

#compute counts by floor bins and transported field
df_floors_tv_n = df_floor_long.drop('passenger_id', axis=1).groupby(
                  ['variable', 'value', 'transported'], as_index=False).size()
df_floors_tv_n = df_floors_tv_n.rename(columns={'value': 'category', 'size': 'count'})

#generate plots
make_grouped_barplot(df=df_floors_tv_n, var='floor')
make_grouped_barplot(df=df_floors_tv_n, var='floor_2', ord=floor_2_lab)
make_grouped_barplot(df=df_floors_tv_n, var='floor_3', ord=floor_3_lab) 
make_grouped_barplot(df=df_floors_tv_n, var='floor_4', ord=floor_4_lab) 
#choose this: fewest categories & large discrepancies
#will need to change order of labels on plot


### Drop remaining floor variables
df_f = df.drop(['floor', 'floor_2', 'floor_3'], axis=1)



# Potential Character String Imputation=============================================================

## Assessment of missingness-------------------------
### Numbers of non-missing vs missing names
df_f['name_present'] = df_f['name'].notna().astype("category")

sns.countplot(x="name_present", data=df_f)
plt.show()
plt.close()


### Name missingness + passenger_group
#### Pull passenger groups containing NA names
df_NA_name = df_f[df_f['l_name'].isna()]
pass_group_NA_name = df_NA_name["passenger_group"].tolist()


#### Calculate numbers of each size of passenger_group and the numbers of named passengers
#select only passenger_groups containing at least one NA l_name
df_pg_w_NA_name = df_f[df_f['passenger_group'].isin(pass_group_NA_name)]

#calculate number of non-NA names out of number of inds in pass group
df_pg_size_name_n = df_pg_w_NA_name.groupby("passenger_group", 
                                            as_index=False).agg(pass_group_size=('l_name', 'size'), 
                                                                   num_name=('l_name', 'count'),
                                                                   diff_names=('l_name', 'nunique'))
df_pg_size_name_n.groupby('diff_names')['diff_names'].value_counts()
#200 with no names; 21 passenger_groups with 2-3 different names so can't impute and 104 with 0
  #so would need to assume that it's a unique l_name (see below)
                                                                   

### Name missingness + same_room
#### Note: two passenger ids where cabin and last name NA (so only 198 NA names where there is a cabin)
df_f[df_f['cabin'].isna() & df_f['l_name'].isna()] #2
df_f[df_f['l_name'].isna()] #200


#### Pull non-NA cabins containing at least one passenger with NA name
t_cabin_filter = df_f[df_f['cabin'].notna() & df_f['l_name'].isna()]['cabin'].tolist()


#### Calculate numbers of each room size (i.e., # of inds) and the numbers of named passengers
df_cabin_filter = df_f[df_f['cabin'].isin(t_cabin_filter)]
df_cabin_nm_room_n = df_cabin_filter.groupby('cabin', 
                                             as_index=False).agg(num_name=('l_name', 'count'),
                                                                 room_group_size=('l_name', 'size'),
                                                                 diff_names=('l_name', 'nunique'))
df_cabin_nm_room_n.groupby('diff_names')['diff_names'].value_counts()
#similar to pg...198 with no l_name and cabin is not NA; of these 122 without any other named
  #passengers and 13 with 2-3 different l_names, so little benefit in imputing 63 names
  #somewhat confidently (see below)
                                                                 
                                                        
### Compare number of distinct l_name values by passenger_group and cabin (all passengers)
#passenger_group
df_f.groupby('passenger_group', as_index=False).agg(diff_names=('l_name', 'nunique')).groupby(
  'diff_names')['diff_names'].value_counts()
#104 passenger_groups that have no named passengers
#249 passenger_groups with 2 or more different last names
#5864 passenger_groups with 1 l_name

#if choose to populate l_name by passenger_group-l_name, then you'd assume that the 0 category are
  #unique names (solo travellers) and would be unable to populate the 2-4 group confidently

#cabin
df_f.groupby('cabin', as_index=False).agg(diff_names=('l_name', 'nunique')).groupby(
  'diff_names')['diff_names'].value_counts()
#122 cabins that have no named passengers
#220 cabins with 2-3 different last names
#6218 cabins with 1 l_name

#similar conclusion as passenger_group


### Check that the same l_name occurs across passenger_groups and/or cabins
#passenger_group
df_f.groupby('l_name').agg(diff_pgs=('passenger_group', 'nunique')).groupby(
 'diff_pgs')['diff_pgs'].value_counts().reset_index()
#this is a breakdown of the instances in which a l_name is found in 1-10 different pgs
#this shows that clearly the same l_name is spread across multiple pgs, so it's impossible to 
  #say if a solo traveler (only traveler with a specific pg) is family of 1 or just 'spillover'
 
#cabin
df_f.groupby('l_name').agg(diff_cabin=('cabin', 'nunique')).groupby(
 'diff_cabin')['diff_cabin'].value_counts().reset_index()
#similar to pgs...same l_name is found in multiple cabins, so unknown if the 122 cabins with no 
  #named passengers are 'families of 1' or not
  
#*ultimately the point of imputing l_name is to quantify family size (as too many l_names for
  #categorical variable) but too many assumptions made; instead, stick with cabin and pg size
  #and see if they are correlated first before proceeding


### Further exploration
#look at transported breakdown for name present/absent
df_f[['name_present', 'transported']].groupby(['name_present', 'transported']).size()
#very similar patterns--almost 50-50

#look at percentage of missingness
(200/8693) * 100 #2.3%

#Decision = no imputation & utilize other information from passengers with missing names



# Create New 'Size' Variables=======================================================================
## Check for NAs-------------------------
df_f[['passenger_group', 'cabin']].isna().sum() 
#0 for pg and 199 for cabin


## Group size-------------------------
df_pg_gs = df_f[['passenger_group']].groupby('passenger_group', as_index=False).agg(
  group_size=('passenger_group', 'size'))


## Cabin size-------------------------
df_cab_cs = df_f[df_f['cabin'].notna()][['cabin']].groupby('cabin', as_index=False).agg(cabin_size=('cabin', 'size'))
df_cab_cs.sum() #8494 (excludes missing values)

## Join them in
df_f_gs = pd.merge(df_f, df_pg_gs, on='passenger_group')
df_fs = pd.merge(df_f_gs, df_cab_cs, on='cabin', how='left')
#s = size


# Data Imputation (numerical and categorical)=======================================================

## Missingness-------------------------
### Assess missingness
df_fs.info()

#by columns
df_fs.isna().sum(axis=0).head(25)

dict_miss = {'index': 'variable', 0: 'n_missing'}
df_fs_report = df_fs.isna().sum(axis=0).sort_values(ascending=False).to_frame().reset_index().rename(
    columns=dict_miss)
df_fs_report=df_fs_report[df_fs_report['n_missing']>0]
#given that so many columns have missing values, assessing visually is difficult so let's use a 
  #statistical test

#by row
df_fs.isna().sum(axis=1).sort_values(ascending=False)


### Test for MCAR
#retain only explanatory variables
df_mcar = df_fs.drop(['passenger_id', 'passenger_group', 'cabin',  'name', 'f_name', 
                      'l_name', 'name_present','num',  'transported'], axis=1)
df_mcar.dtypes
df_mcar.isna().sum(axis=0).head(20)

#run test
littles_mcar_test(df_mcar)
#X^2 = 8.46 x 10^-30 and p = 1; thus MCAR



## Handling missing data----------------------------------------
#prefer to retain as much data as possible and to use a nuanced imputation technique, so am going
  #with multiple imputation

### Prepare data for imputation
#### Remove extraneous columns
cols_to_drop = ['passenger_id', 'passenger_group', 'cabin',  'name', 'f_name', 'l_name', 
                'num', 'transported', 'name_present']

df_fs_imp = df_fs.drop(cols_to_drop, axis=1)


### Convert bool to categorical and split data by data type
#bool to category
df_fs_imp[['cryo_sleep', 'vip']] = df_fs_imp[['cryo_sleep', 'vip']].astype('category')

#create lists of data types
cats = ['ticket', 'home_planet', 'deck', 'side', 'destination', 'floor_4']
# cats.append('floor_4') #['ticket', 'home_planet', 'deck', 'side', 'destination', 'floor_4']
nums = df_fs_imp.columns.drop(cats).tolist() # includes cryo_sleep & vip and group_size & cabin_size

#split data
df_imp_nums = df_fs_imp[nums]
df_imp_cats = df_fs_imp[cats]


### Impute data using median for numerical columns and most frequent for categorical ones
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


### Check for missingness
df_fs_imp.info()
#columns with missing values: cabin, name, f_name, l_name, num



# Save Data to File=================================================================================
#save in pickle format to retain data types and categories
# afile = open('train_impute.pkl', 'wb')
# pickle.dump(df_fs_imp, afile)
# afile.close()



