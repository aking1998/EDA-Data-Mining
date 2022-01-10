"""
Question 1: Simple linear regression
The file ./specs/marks_question1.csv contains data about midterm and final exam grades for
a group of students.
1. Plot the data using matplotlib. Do midterm and final exam seem to have a linear relationship?
Discuss the data and their relationship in your report. Save your plot to ./output/marks.png.
2. Use linear regression to generate a model for the prediction of a students’ final exam grade based
on the students’ midterm grade in the course, then describe the model in your report.
3. According to your model, what will be the final exam grade of a student who received an 86 on
the midterm exam?
"""

# Importing python libraries

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Reading Dataset

marks_df = pd.read_csv("./specs/marks_question1.csv")
print(marks_df.head)

# Converting to Numpy array

x = np.array([marks_df['midterm']])
y = np.array([marks_df['final']])

# Plotting a Scatter Plot

plt.xlabel('midterm')
plt.ylabel('final')
plt.scatter(x, y, color='red')
plt.show()

# Reshaping the array

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# Fitting a linear regresssion model

reg_model = LinearRegression()

reg_model.fit(x, y)

y_pred = reg_model.predict(x)

# Calculating slope, intercept, root mean squared error and R2 score 

rmse = mean_squared_error(y, y_pred)
R2 = r2_score(y, y_pred)


print('Slope:' ,reg_model.coef_)
print('Intercept:', reg_model.intercept_)
print('Root mean squared error: ', rmse)
print('R2 score: ', R2)

# plotting the linear relationship between the two variables

plt.scatter(x, y, s=10)
plt.xlabel('midterm')
plt.ylabel('final')


plt.plot(x, y_pred, color='r')
plt.savefig('./output/marks.png')
plt.show()

# predicting final marks of student who scored 86 in midterm exam

print("The marks of Student for final exam who scored 86 in midterm exam is ")
print(reg_model.predict([[86]]))


"""
Question 2: Classification with Decision Tree
The file ./specs/borrower_question2.csv contains bank data about customers (borrowers) that
may or may not have being defaulted.
1. Filter out the TID attribute, as it is not useful for decision making.
2. Using sklearn decision trees, generate a decision tree using information gain as splitting
criterion, and a minimum impurity decrease of 0.5. Leave everything else to its default
value. Plot the resulting decision tree, and discuss the classification results in your report.
Save the produced tree into ./output/tree_high.png.
3. Train another tree, but this time use a minimum impurity decrease of 0.1. Plot the
resulting decision tree, and compare the results with the previous model you trained. Save
the produced tree into ./output/tree_low.png.
"""

# Importing python libraries

import pandas as pd
import graphviz
from sklearn import tree

# Reading Dataset

df = pd.read_csv("./specs/borrower_question2.csv")
print(df.head)

# Removing TID attribute from the dataset

df = df.drop(columns=['TID'])
df

# Splitting the data into input and target data and normalising the attributes

inputs = df.drop('DefaultedBorrower',axis='columns')
target_data = df['DefaultedBorrower']

input_data = pd.get_dummies(inputs)
print(input_data.head)

# Fitting the decision tree model with criteria = "entropy" and minimum impurity decrease = 0.5

high_model = tree.DecisionTreeClassifier(criterion='entropy',min_impurity_decrease=0.5)
high_model.fit(input_data, target_data)
hscore = high_model.score(input_data,target_data)
print(hscore)


# Visualising the decision tree model using graphviz

dot_data = tree.export_graphviz(high_model, out_file=None,feature_names=input_data.columns,class_names=target_data,filled=True, rounded=True,special_characters=True)  
high_graph = graphviz.Source(dot_data, format="png") 

high_graph.render("./output/tree_high.png")

# Fitting the decision tree model with criteria = "entropy" and minimum impurity decrease = 0.1

low_model = tree.DecisionTreeClassifier(criterion='entropy',min_impurity_decrease=0.1)
low_model.fit(input_data, target_data)
lscore = low_model.score(input_data, target_data)
print(lscore)

# Visualising the decision tree model using graphviz

dot_data = tree.export_graphviz(low_model, out_file=None,feature_names=input_data.columns,class_names=target_data,filled=True, rounded=True,special_characters=True)  
low_graph = graphviz.Source(dot_data,format="png") 

low_graph.render("./output/tree_low.png")













