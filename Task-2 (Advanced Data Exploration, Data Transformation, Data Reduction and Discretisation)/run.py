"""
Question 1: Advanced Data Exploration
A module coordinator has just completed the module assessments, and s/he would like to
perform a quick analysis on the students results in various components of the module. The
main objective is to see if there is any correlation between the assessment components. The
students’ results are given in the file “Students_Results.csv”. Using Python script, answer the
following questions:
1. Find the minimum, maximum, mean and standard deviation for each Homework column
and the exam column.
2. Add an additional named as ‘Homework Avg’ for the average homework mark for each
student. Assume that the weighting of the homework average is 25% and that of the
examination is 75%, add an additional column named 'Overall Mark' for the overall folded
mark.
3. Construct a correlation matrix of homework and exam variables. What can you conclude
from the matrix?
4. Discuss various ways of treating the missing values in the dataset.
5. Use UCD grading system to convert the final mark into a grade (column named ‘Grade’).
Produce a histogram for the grades.
6. Save the newly generated dataset to “./output/question1_out.csv”.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('./specs/Students_Results.csv')
data

data["Homework 1"] = data["Homework 1"].replace(np.NaN, data["Homework 1"].mean())          # Replacing Missing field with average
data["Homework 2"] = data["Homework 2"].replace(np.NaN, data["Homework 2"].mean())
data["Homework 3"] = data["Homework 3"].replace(np.NaN, data["Homework 3"].mean())
data["Exam"] = data["Exam"].replace(np.NaN, data["Exam"].mean())
print(data["Homework 1"][:54], data["Homework 2"][:54], data["Homework 3"][:54], data["Exam"][:54])


Min_hw1 = data["Homework 1"].min()                       # Calculating Minimum
Min_hw2 = data["Homework 2"].min()
Min_hw3 = data["Homework 3"].min()
Min_Exam = data.Exam.min()
print("Homework 1 Minimum is : ", Min_hw1)
print("Homework 2 Minimum is : ", Min_hw2)
print("Homework 3 Minimum is : ", Min_hw3)
print("Exam Minimum is : ", Min_Exam)

Max_hw1 = data["Homework 1"].max()                        # Calculating Maximum
Max_hw2 = data["Homework 2"].max()
Max_hw3 = data["Homework 3"].max()
Max_Exam = data.Exam.min()
print("Homework 1 Maximum is : ", Max_hw1)
print("Homework 2 Maximum is : ", Max_hw2)
print("Homework 3 Maximum is : ", Max_hw3)
print("Exam Maximum is : ", Max_Exam)


Mean_hw1 = data["Homework 1"].mean()                       # Calculating Mean
Mean_hw2 = data["Homework 2"].mean()
Mean_hw3 = data["Homework 3"].mean()
Mean = data.Exam.mean()
print("Homework 1 Mean is : ", Mean_hw1)
print("Homework 2 Mean is : ", Mean_hw2)
print("Homework 3 Mean is : ", Mean_hw3)
print("Exam Mean is : ", Mean)

hw1_std = data["Homework 1"].std()                          # Calculating Standard Deviation
hw2_std = data["Homework 2"].std()
hw3_std = data["Homework 3"].std()
exam_std = data["Exam"].std()
print("The standard deviation of the Homework 1 column is:", hw1_std)
print("The standard deviation of the Homework 2 column is:", hw2_std)
print("The standard deviation of the Homework 3 column is:", hw3_std)
print("The standard deviation of the Exam column is:", exam_std)

df = pd.DataFrame(data,columns=['Homework 1','Homework 2','Homework 3','Exam'])

corrMatrix = df.corr()      # Calculating Correlation Matrix
print (corrMatrix)


data['Homework Avg'] = data[['Homework 1','Homework 2','Homework 3']].mean(axis=1)
data['Homework Avg'] = round(data['Homework Avg'],1)

data["Overall Mark"] = np.average(data[['Homework Avg','Exam']],weights=[0.25,0.75],axis=1)
data["Overall Mark"] = round(data["Overall Mark"],2)


data['Grade'] = (np.where((data['Overall Mark'] >= 90) & (data['Overall Mark'] <= 100),
                          'A+',
                np.where((data['Overall Mark'] >= 80) & (data['Overall Mark'] < 90),
                         'A',
                np.where((data['Overall Mark'] >= 70) & (data['Overall Mark'] < 80),
                         'A-',
                np.where((data['Overall Mark'] >= 66.67) & (data['Overall Mark'] < 70),
                         'B+',
                np.where((data['Overall Mark'] >= 63.33) & (data['Overall Mark'] < 66.67),
                         'B',        
                np.where((data['Overall Mark'] >= 60) & (data['Overall Mark'] < 63.33),
                         'B-',         
                np.where((data['Overall Mark'] >= 56.67) & (data['Overall Mark'] < 60),
                         'C+',         
                np.where((data['Overall Mark'] >= 53.33) & (data['Overall Mark'] < 56.67),
                         'C',         
                np.where((data['Overall Mark'] >= 50) & (data['Overall Mark'] < 53.33),
                         'C-',
                np.where((data['Overall Mark'] >= 46.67) & (data['Overall Mark'] < 50),
                         'D+',
                np.where((data['Overall Mark'] >= 43.33) & (data['Overall Mark'] < 46.67),
                         'D',
                np.where((data['Overall Mark'] >= 40) & (data['Overall Mark'] < 43.33),
                         'D-',        
                np.where((data['Overall Mark'] >= 36.67) & (data['Overall Mark'] < 40),
                         'E+',         
                np.where((data['Overall Mark'] >= 33.33) & (data['Overall Mark'] < 36.67),
                         'E',         
                np.where((data['Overall Mark'] >= 30) & (data['Overall Mark'] < 33.33),
                         'E-',         
                np.where((data['Overall Mark'] >= 26.67) & (data['Overall Mark'] < 30),
                         'F+',
                np.where((data['Overall Mark'] >= 23.33) & (data['Overall Mark'] < 26.67),
                         'F',
                np.where((data['Overall Mark'] >= 20) & (data['Overall Mark'] < 23.33),
                         'F-',
                np.where((data['Overall Mark'] >= 16.67) & (data['Overall Mark'] < 20),
                         'G+',
                np.where((data['Overall Mark'] >= 13.33) & (data['Overall Mark'] < 16.67),
                         'G',        
                np.where((data['Overall Mark'] >= 0.01) & (data['Overall Mark'] < 13.33),
                         'G-','No work was submitted by the student or the student was absent from assessment'         
                        
               ))))))))))))))))))))))         
print(data)



data["Grade"].hist(bins=10)                 #Plotting Histogram for 'Grade' column

data.to_csv('./output/question1_out.csv', index=False)

"""
Question 2: Data Transformation
The file “Sensor_Data.csv” contains data obtained from a sensory system. Some of the
attributes in the file need to be normalised, but you don't want to lose the original values.
1. Generate a new attribute called “Original Input3” which is a copy of the attribute “Input3”.
Do the same with the attribute “Input12” and copy it into Original “Input12”.
2. Normalise the attribute “Input3” using the z-score transformation method.
3. Normalise the attribute “Input12” in the range [0:0; 1:0].
4. Generate a new attribute called “Average Input”, which is the average of all the attributes
from “Input1” to “Input12”. This average should include the normalised attributes values
but not the copies that were made of these.
5. Save the newly generated dataset to “./output/question2_out.csv”.

"""

df = pd.read_csv("./specs/Sensor_Data.csv")


df["Original Input3"]=df["Input3"]
df["Original Input12"]=df["Input12"]
print(df)

from scipy.stats import zscore                           # Normalization using z-score
df['Input3'] = zscore(df['Input3'])
print(df)


from sklearn.preprocessing import MinMaxScaler

x = df[["Input12"]] #returns a numpy array                # Normalization using MinMaxScaler
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df[["Input12"]] = pd.DataFrame(x_scaled)
df["Input12"] = round(df["Input12"],3)


df['Average Input'] = df[['Input1','Input2','Input3','Input4','Input5','Input6','Input7','Input8','Input9','Input10','Input11','Input12']].mean(axis=1)
print(df)

df.to_csv('./output/question2_out.csv', index=False)


"""
Question 3: Data Reduction and Discretisation
The files “DNA_Data.csv” contains biological data arranged into multiple columns. We need
to compress the information contained in the data.
1. Reduce the number of attributes using Principal Component Analysis (PCA), making sure
at least 95% of all the variance is explained.
2. Discretise the PCA-generated attribute subset into 10 bins, using bins of equal width. For
each component X that you discretise, generate a new column in the original dataset
named “pcaX_width”. For example, the first discretised principal component will
correspond to a new column called “pca1_width”.
3. Discretise PCA-generated attribute subset into 10 bins, using bins of equal frequency (they
should all the same number of points). For each component X that you discretise, generate
a new column in the original dataset named “pcaX_freq”. For example, the first discretised
principal component will correspond to a new column called “pca1_freq”.
4. Save the generated dataset to “./output/question3_out.csv”.
""" 

df = pd.read_csv("./specs/DNA_Data.csv")


"""from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)                                           # Normalizing the data
scaled_data = scaler.transform(df)
print(scaled_data)
"""

from sklearn.decomposition import PCA                    # Applying PCA on data
pca = PCA(n_components = 0.95)
pca.fit(df)
pca_df = pd.DataFrame(pca.transform(df))



for i in pca_df.columns:
    df["pca{}_width".format(i)] = pd.cut(pca_df[i], 10)     # Applying Binning Techniques
    df["pca{}_freq".format(i)] = pd.qcut(pca_df[i],10)

print(df.shape)


df.to_csv('./output/question3_out.csv', index=False)


