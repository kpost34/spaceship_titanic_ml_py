#Functions to help with coding
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# EDA Function======================================================================================
def make_grouped_barplot(df, var):
  df_var_plot = df[df['variable'] == var]
  p_var = sns.catplot(x="category", y="count", kind="bar", hue="transported", data=df_var_plot)
  p_var.set_axis_labels(x_var=var.title(), y_var="Number")
  plt.show()
  plt.close()
  
  
# Data Wrangling Functions==========================================================================
## Function that simulate a case_when() [archive?]
def name_conversion(row):
    if pd.notna(row['name']):
        return row['name']
    elif pd.isna(row['f_name']) and pd.notna(row['l_name']):
        return row['f_name'] + ' ' + row['l_name']
    elif pd.isna(row['f_name']) and pd.isna(row['l_name']):
        return ""
    else:
        return "CHECK"
  
  
## Function for Little's MCAR test
def littles_mcar_test(data):
  # Calculate the number of missing values in each column
  missing_counts = data.isnull().sum()
  
  # Remove columns with zero missing values
  missing_counts = missing_counts[missing_counts > 0]
  
  if len(missing_counts) == 0:
      print("No missing values found. Unable to perform Little's MCAR test.")
      return
  
  # Calculate the total number of missing values
  total_missing = missing_counts.sum()
  
  # Compute the expected frequencies for each column
  expected_freqs = (missing_counts / total_missing).values
  
  # Perform chi-square test of goodness of fit
  chi2_stat, p_val, _, _ = chi2_contingency(np.array([missing_counts.values, expected_freqs * total_missing]))
  
  # Print the result
  print("Chi-square statistic:", chi2_stat)
  print("P-value:", p_val)
  
  # Interpret the result
  if p_val < 0.05:
      return "Reject the null hypothesis: Missingness is not completely at random."
  else:
      return "Fail to reject the null hypothesis: Missingness is completely at random."


# Feature Engineering Functions=====================================================================
## Function to create count plots of each categorical variable
def make_countplots(df, var):
  df_var_plot = df[df['variable'] == var]
  p_var = sns.catplot(x="category", y="count", kind="bar", hue="transported", data=df_var_plot)
  p_var.set_axis_labels(x_var=var.title(), y_var="Number")
  plt.show()
  plt.close()


## Function to group rare categories
def group_categories(variable, rare_cats, new_cat):
  if variable in rare_cats:
    return new_cat
  else:
    return variable




