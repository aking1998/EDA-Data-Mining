"""
Question 1: Association rules with Apriori
The file “gpa_question1.csv” contains a data sample of University students. The main
objective of this exercise is to extract interesting association rules, if there is any, from this
file. As in any data mining process, we start by pre-processing the dataset before its analysis.
1. Preprocess the data (if need) so that it is free from missing values, noise, outliers.
2. Use the Apriori algorithm to generate frequent itemsets from the input data, with a
minimum support equals to 0.15. In your answer, comment on the number of frequent
itemsets and their sizes.
3. Does the attribute “count” have an impact on the Apriori algorithm’s results? Justifier
your answer.
4. Sort the itemsets according to the support in descending order. Save the generated
itemsets into ./output/question1_out_apriori.csv. Include the support column in your
output file.
5. Using these frequent itemsets, find all association rules with a minimum confidence
equals to 0.6.
6. Sort the itemsets according to the confidence in descending order. Save the generated
rules into ./output/question1_out_rules06.csv. Include the support and confidence
columns in your output file.
7. Using the same frequent itemsets as in 5), find all association rules that satisfy a
minimum confidence of 0.9. Include a short description for major 5 rules in your
report.
8. Sort the itemsets according to the confidence in descending order. Save the generated
rules into ./output/question1_out_rules09.csv in the same format as in the previous
questions.
"""

# Importing Libaries

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth

# Reading the Datset
df = pd.read_csv('./specs/gpa_question1.csv')


df = df.loc[np.repeat(df.index.values,df['count'])]


df.drop(columns=['count'], axis = 0, inplace=True)


dataset = df.to_numpy()

# Applying Transaction encoder on th dataset
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
dataset = pd.DataFrame(te_ary, columns = te.columns_)


# Applying Apriori Algorithm and generating frequent itemsets
frequent_itemsets = apriori(dataset, min_support = 0.15, use_colnames = True)



frequent_itemsets = frequent_itemsets.sort_values(by=['support','itemsets'],axis=0, ascending=False)


frequent_itemsets.to_csv('./output/question1_out_apriori.csv', index=False)

# Generation of association rules with confidence = 60%
res = association_rules(frequent_itemsets, metric='confidence', min_threshold = 0.6)
print(res)


res1 = res.sort_values(by=['confidence'],axis=0, ascending=False)


res1.to_csv('./output/question1_out_rules06.csv', index=False)

# Generation of association rules with confidence = 90%
res2 = association_rules(frequent_itemsets, metric='confidence', min_threshold = 0.9)


res2 = res2.sort_values(by=['confidence'],axis=0, ascending=False)


res2.to_csv('./output/question1_out_rules09.csv', index=False)

"""
Question 2: Association rules with FP-Growth
The file ./specs/bank_data_question2.csv contains customer records from the marketing
department of a financial firm. The data contains the following fields.
id: a unique identification number of a customer
age: age of customer in years (numeric)
sex: customer’s gender (MALE or FEMALE)
region: inner city / rural / suburban / town
income: income of customer (numeric)
married: YES / NO
children: number of children (numeric)
car: owns a car  YES / NO
save acct: if the customer has a saving account - YES / NO
current acct: if the customer has a current account - YESY / NO
mortgage: if the customer has a mortgage - YES / NO
pep: if the customer has signed for a Personal Equity Plan after the last
mailing - YES / NO
1. Data Pre-processing:
a. Which attributes should be selected for data mining task? Justify your answer.
(Hint: explain why you exclude, if any, some attributes).
b. Discretize the numeric attributes into 3 bins of equal width.
2. Assume that the minimum support is equal to 20%. Use the FP-Growth algorithm to
generate frequent all frequent itemsets. Comment in your report on the frequent
itemsets, number, size, and usefulness.
3. Sort the itemsets according to the support in descending order. Save the generated
itemsets into ./output/question2_out_fpgrowth.csv
4. Generate all the rules associated to these frequent itemsets.
5. Which confidence values that can return a set of rules of size at least equal to 10 (i.e.,
the number of rules). Explain in your report how did you identify these confidence
values.
6. Sort the itemsets according to the confidence in descending order. Save the generated
rules into ./output/question2_out_rules.csv
7. Identify 4 most interesting rules, explaining, for each rule:
 why you believe it is interesting, based on the company’s business objectives,
 the recommendations that might help the company to better understand
behaviour of its customers or its marketing campaign.
Note that the most interesting rules should provide some non-trivial and actionable
knowledge based on the underlying business objectives.
"""


df = pd.read_csv('./specs/bank_data_question2.csv')

# Discretizing the numeric attributes into 3 bins using the cut() method.
df['age'] = pd.cut(df['age'],3, labels = ['age(34.333, 50.667)','age(50.667, 67.0)','age(17.951, 34.333)'])
df['income'] = pd.cut(df['income'],3, labels =['income(4956.094, 24386.173)','income(24386.173, 43758.137)','income(43758.137, 63130.1)'])
df['children'] = pd.cut(df['children'],3, labels = ['children(0_1)','children(1_2)','children(2_3)'])


df.drop(columns=['id'], axis = 0, inplace=True)


df['married'] = 'Married_' + df['married']
df['car'] = 'Car_' + df['car']
df['save_act'] = 'save_act_ ' + df['save_act']
df['current_act'] = 'current_act_ ' + df['current_act']
df['mortgage'] = 'mortgage_' + df['mortgage']
df['pep'] = 'pep_' + df['pep']

print(df)


dataset = df.to_numpy()

# Applying Transaction encoder on th dataset
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
dataset = pd.DataFrame(te_ary, columns = te.columns_)
print(dataset.columns)


# Applying FP Growth Algorithm and generating frequent itemsets
frequent_itemsets = fpgrowth(dataset, min_support = 0.20, use_colnames = True)

print(frequent_itemsets.shape)

frequent_itemsets = frequent_itemsets.sort_values(by=['support'],axis=0, ascending=False)
frequent_itemsets.to_csv('./output/question2_out_fpgrowth.csv', index=False)


# Generation of association rules
res = association_rules(frequent_itemsets, metric='confidence', min_threshold = 0)

res3 = res.sort_values(by=['confidence'],axis=0, ascending=False)
res3.to_csv('./output/question2_out_rules.csv', index=False)




