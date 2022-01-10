#!/usr/bin/env python
# coding: utf-8

"""
Question 1: Data Exploration
Suppose that the data for analysis includes the attribute age. The age values for the data
tuples are (in increasing order):
13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25, 25, 30, 33, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70.
1- What is the mean of the data? What is the median?
2- What is the mode of the data? Comment on the data’s modality (i.e., bimodal, trimodal, etc.).
3- What is the midrange of the data?
4- Can you find (roughly) the first quartile (Q1) and the third quartile (Q3) of the data?
5- Give the five-number summary of the data.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot
a = np.array([13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25, 25, 30, 33, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70])
print("Mean is : ",np.mean(a))                   # Mean

print("Median is :",np.median(a))                # Median

mode = stats.mode(a)
print("Mode is : ",mode)                         # Mode

midrange=np.min(a)+np.max(a)/2
print("Midrange is : ",midrange)                 # Midrange


print("Q1 or 25th percentile :   ", np.percentile(a, 25))
print("Q2 or 75th percentile :   ", np.percentile(a, 75))      # Quartile 1 and Quartile 3

print("Five Number Summary : ")
print("Minimum :    ", a.min())
print("25th percentile or Q1:   ", np.percentile(a, 25))
print("Median:            ", np.median(a))
print("75th percentile or Q3:   ", np.percentile(a, 75))       # Five Point Summary
print("Maximum :    ", a.max())

matplotlib.pyplot.boxplot(a, notch=None, vert=None, patch_artist=None, widths=None)      # Boxplot

"""
Question 2: Data Cleaning
The file AutoMpg question1.csv contains data related to cars, such as horsepower, weight, car name,
and so on. Unfortunately, some of the values for the horsepower and origin columns were not
properly recorded. Can you tell how many missing values are there for each one of these columns?
Write the answer in your report.
1. Replace the missing horsepower values with the average of this column.
2. Replace the missing origin values with the minimum of this column
3. Save the generated data file to ./output/question1 out.csv
When saving the generated data, pay extra attention to the columns included in the file (hint: if you
are using pandas, take a look at the arguments of the to_csv function).
"""
import pandas as pd
data = pd.read_csv('./specs/AutoMpg_question1.csv')               # Reading dataset
data

print(" \nCount total NaN at horsepower column is :",
      data.horsepower.isnull().sum())                     # Counts missing values for horsepower column

print(" \nCount total NaN at origin column is :",
      data.origin.isnull().sum())                         # Counts missing values for origin column

data['horsepower'] = data['horsepower'].fillna((data['horsepower'].mean()))           
data

data['origin'] = data['origin'].fillna((data['origin'].min()))
data

data.to_csv('./output/question1_out.csv', index=False)

"""
Question 3: Data Integration
The files AutoMpg question2 a.csv and AutoMpg question2 b.csv contain similar pieces of information
about car models. There are some differences between the 2 files. What you need to do is:
1. The dataset A has an attribute called car name, whereas the dataset B has an attribute called
name. Rename the name attribute to car name (unintended tongue twister!).
2. The dataset B has an attribute called other, which is not present in the dataset A. Create an
attribute called other in the dataset A and assign it a default value of 1.
3. Concatenate dataset A and B together, and just like in question 1, save the resulting file to
./output/question2 out.csv.
"""
import pandas as pd
data_1 = pd.read_csv("./specs/AutoMpg_question2_a.csv")
data_1

data_2 = pd.read_csv("./specs/AutoMpg_question2_b.csv")
data_2


data_2 = data_2.rename(columns={"name":"car name"})              # Renaming column name
data_2

data_1['other'] = '1'                                            # Adding new column with default value
data_1

result = pd.concat([data_1, data_2])       # Concatenating the two datasets
result

result.to_csv('./output/question2_out.csv', index=False)
