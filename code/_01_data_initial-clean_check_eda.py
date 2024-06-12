# This script does initial data cleaning/wrangling and performs EDA


# Load Libraries, Set Options, and Change WD========================================================
## Load libraries
import pandas as pd
import inflection
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
from _00_helper_fns import make_grouped_barplot


## Data
os.chdir(root + 'data') 
df0 = pd.read_csv("train.csv") 
#convert to snake_case
df0.columns = df0.columns.map(inflection.underscore) 



# Explore Data======================================================================================
## Get basic info
df0.shape #(8693, 14)
df0.columns
df0.dtypes #nearly all are objects or float64s except for Transported (bool)

df0.head() 
df0.describe()

df0.passenger_id.head()
df0.cabin.head()
df0.name.head() 


## Assess missingness
#missing cases per row
df0.dropna() #6606 rows with 0 NAs
df0.dropna(thresh=1) #8693 rows with 0-1 NA
6693-6606 #87 rows with 1 NA

#missing cases per variable
df0.isna().sum().sort_values(ascending=False) #NAs by column
#transported (DV) and passenger_id (ID) have 0
#remaining vars have 179-217


# Initial Wrangling=================================================================================
## Copy DF
df = df0.copy()


## Split objects into multiple columns
df[['passenger_group', 'ticket']] = df.passenger_id.str.split('_', n=1, expand=True)
df[['deck', 'num', 'side']] = df.cabin.str.split('/', n=2, expand=True)
df[['f_name', 'l_name']] = df.name.str.split(' ', n=1, expand=True) 

df.head() 
df.dtypes


## Convert subset of objects to categorical variables
cats = ['ticket', 'home_planet', 'deck', 'side', 'destination']
df[cats] = df[cats].astype('category')
df[cats].dtypes 


## Convert subset of objects to boolean variables
bools = ['cryo_sleep', 'vip']
df[bools] = df[bools].astype('bool')
df[bools].dtypes


## Create long versions of data focused on categorical and numerical variables
#categorical variables
df_long = pd.melt(df, id_vars=["passenger_id"])
df_cats = df_long.loc[df_long["variable"].isin(cats)] 
df_cats_tv = df_cats.merge(df[['passenger_id', 'transported']], how="left", on="passenger_id")

#boolean variables
df_bools = df_long.loc[df_long["variable"].isin(bools)]
df_bools_tv = df_bools.merge(df[['passenger_id', 'transported']], how="left", on="passenger_id")

#numerical variables
nums = ["age", "room_service", "food_court", "shopping_mall", "spa", "vr_deck"]
df_nums = df_long.loc[df_long["variable"].isin(nums)]
df_nums_tv = df_nums.merge(df[['passenger_id', 'transported']], how="left", on="passenger_id")



# Exploratory Data Analysis=========================================================================
## Univariate plots-------------------------
### Numerical data (histograms)---------------
#with matplotlib
df.describe()

fig, axes = plt.subplots(2, 3)

axes[0, 0].hist(df.age, color="blue")
axes[0, 1].hist(df.room_service, color="green") 
axes[0, 2].hist(df.food_court, color="red") 
axes[1, 0].hist(df.shopping_mall, color="purple") 
axes[1, 1].hist(df.spa, color="black") 
axes[1, 2].hist(df.vr_deck, color="grey") 

axes[0, 0].set(xlabel="age", ylabel="count")
axes[0, 1].set(xlabel="room_service", ylabel="count")
axes[0, 2].set(xlabel="food_court", ylabel="count")
axes[1, 0].set(xlabel="shopping_mall", ylabel="count")
axes[1, 1].set(xlabel="spa", ylabel="count")
axes[1, 2].set(xlabel="vr_deck", ylabel="count")

plt.show() 
plt.close() 


### Categorical data (barplots of counts)---------------
fig, axes = plt.subplots(2, 3)

axes[0, 0].bar(x=df['ticket'].value_counts().index,
               height=df['ticket'].value_counts(), color="blue")
axes[0, 1].bar(df['home_planet'].value_counts().index,
               height=df['home_planet'].value_counts(), color="green") 
axes[0, 2].bar(x=df['deck'].value_counts().index,
               height=df['deck'].value_counts(), color="red")
axes[1, 0].bar(df['side'].value_counts().index,
               height=df['side'].value_counts(), color="purple") 
axes[1, 1].bar(df['destination'].value_counts().index,
               height=df['destination'].value_counts(), color="black") 
axes[1, 2].axis('off')

axes[0, 0].set(xlabel="ticket", ylabel="count")
axes[0, 1].set(xlabel="home_planet", ylabel="count")
axes[0, 2].set(xlabel="deck", ylabel="count")
axes[1, 0].set(xlabel="side", ylabel="count")
axes[1, 1].set(xlabel="destination", ylabel="count")

plt.show()
plt.close()


### Boolean data (barplots of counts)---------------
fig, axes = plt.subplots(1, 2)

axes[0].bar(x=df['cryo_sleep'].value_counts().index,
               height=df['cryo_sleep'].value_counts(), color="blue")
axes[1].bar(x=df['vip'].value_counts().index,
               height=df['vip'].value_counts(), color="green") 
               
axes[0].set(xlabel="cryo_sleep", ylabel="count")
axes[1].set(xlabel="vip", ylabel="count")
               
plt.show()
plt.close()


## Bivariate plots (predictor-target)-------------------------
### Numerical data (boxplots)---------------
#linear scale
sns.catplot(x="transported", y="value", kind="box", row="variable", hue='transported',
            data=df_nums_tv, sharey=False)
plt.show()
plt.close()
#all but age need a pseudo-log scale

#try log scale
p = sns.catplot(x="transported", y="value", kind="box", row="variable", hue='transported',
            data=df_nums_tv, sharey=False)
p.set(yscale="log")
plt.show()
plt.close()
#unsure if that's right...let's check

#group by transported then look at distribution
df[['room_service', 'transported']].groupby('transported').describe()
df[['spa', 'transported']].groupby('transported').describe()
#shows that log scale is displaying correctly


### Categorical data (grouped bars)---------------
#### Data wrangling to get into form
cats_tv = cats.copy()
cats_tv.append('transported')

df_cats_tv_n = df_cats_tv.drop('passenger_id', axis=1).groupby(['variable', 'value', 'transported'], 
                                                               as_index=False).size()
df_cats_tv_n = df_cats_tv_n.rename(columns={'value': 'category', 'size': 'count'})


#### Hard coded with ticket
df_ticket_plot = df_cats_tv_n[df_cats_tv_n['variable'] == 'ticket']

p_ticket = sns.catplot(x="category", y="count", kind="bar", hue="transported", data=df_ticket_plot)
p_ticket.set_axis_labels(x_var="Ticket", y_var="Number")
plt.show()
plt.close()


#### Create plots
make_grouped_barplot(df=df_cats_tv_n, var='ticket')
make_grouped_barplot(df=df_cats_tv_n, var='home_planet')
make_grouped_barplot(df=df_cats_tv_n, var='deck')
make_grouped_barplot(df=df_cats_tv_n, var='side')
make_grouped_barplot(df=df_cats_tv_n, var='destination')


### Boolean data (grouped bars)---------------
#### Data wrangling to get into form
bools_tv = bools.copy()
bools_tv.append('transported')

df_bools_tv_n = df_bools_tv.drop('passenger_id', axis=1).groupby(['variable', 'value', 'transported'], 
                                                                 as_index=False).size()
df_bools_tv_n = df_bools_tv_n.rename(columns={'value': 'category', 'size': 'count'})


#### Create plots
make_grouped_barplot(df=df_bools_tv_n, var='cryo_sleep')
make_grouped_barplot(df=df_bools_tv_n, var='vip')


# Save Data to File=================================================================================
# df.to_csv('train_initial_clean.csv', index=False)

#save in pickle format to retain data types and categories
afile = open('train_initial_clean.pkl', 'wb')
pickle.dump(df, afile)
afile.close()




