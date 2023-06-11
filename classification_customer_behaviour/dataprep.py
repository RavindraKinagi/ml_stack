# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Load the dataset
df = pd.read_csv('../input/customer-behaviour/Customer_Behaviour.csv')
df.head()

# Data exploration and visualization
df.shape
df.size
df.describe()
sns.histplot(df['EstimatedSalary'], kde=True)
sns.scatterplot(df['EstimatedSalary'], df['Purchased'])
sns.boxenplot(df['EstimatedSalary'])
mean = df['EstimatedSalary'].mean()
std = df['EstimatedSalary'].std()
df['Z-score'] = df['EstimatedSalary'] - mean/3*std
df.head()
sns.histplot(df['Z-score'], kde=True)
color = ('red', 'green')
explode = [0.01, 0.01]
df['Gender'].value_counts().plot(kind='pie', colors=color, explode=explode)
sns.scatterplot(df['Age'], df['EstimatedSalary'], color='Red')
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
corr = df.corr()
style.use('ggplot')
sns.heatmap(corr, cmap='bwr', annot=True)

# Split the data
x = df[['Gender', 'EstimatedSalary']]
y = df[['Purchased']]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=7)
xtrain.shape, xtest.shape, ytrain.shape, ytest.shape

# Model training and evaluation
rf = RandomForestClassifier()
rf.fit(xtrain, ytrain)
rf_predict = rf.predict(xtest)
accuracy_score(rf_predict, ytest)

nb = GaussianNB()
nb.fit(xtrain, ytrain)
nb_predict = nb.predict(xtest)
accuracy_score(nb_predict, ytest)

tree = DecisionTreeClassifier()
tree.fit(xtrain, ytrain)
tree_predict = tree.predict(xtest)
accuracy_score(tree_predict, ytest)

