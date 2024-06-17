#This script contains functions to help with coding

# Load Packages=====================================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
# from IPython.display import display, HTML
import datatable as dt
import IPython.display as ipd


# EDA Function======================================================================================
def make_grouped_barplot(df, var, ord=None, rot=0):
  df_var_plot = df[df['variable'] == var]
  p_var = sns.catplot(x="category", y="count", kind="bar", hue="transported", data=df_var_plot,
                      order=ord)
  p_var.set_axis_labels(x_var=var.title(), y_var="Number")
  p_var.set_xticklabels(rotation=rot)
  
  p_var=sns.move_legend(p_var, loc="lower center", bbox_to_anchor=(.5, -.18), ncol=2)

  plt.show()
  plt.close()
  
  
  
# Data Wrangling Function===========================================================================
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



# Report Functions==================================================================================
## DT-related functions
### Function to generate a simple DT
def create_simple_dt(df, head=True):
  # Convert the pandas dataframe to a datatable Frame
  dt_frame = dt.Frame(df)

  # Remove row numbers/indexes
  dt_frame.key = None

  # Convert datatable Frame to pandas DataFrame for HTML rendering
  pd_df = dt_frame.to_pandas()

  # Generate HTML without index
  html = pd_df.to_html(index=False, header=head)

  return ipd.HTML(html)


