# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

# Reading the dataset
df = pd.read_csv('/kaggle/input/water-quality/waterQuality1.csv')

# Checking the first few rows of the dataset
df.head()

# Checking the information of the dataset
df.info()

# Checking the data types of columns
df.dtypes

# Checking the count of null values in each column
df.isnull().sum()

# Replacing '#NUM!' with NaN values in 'ammonia' column
df['ammonia'] = df['ammonia'].replace('#NUM!', pd.np.nan)
df['ammonia'] = df['ammonia'].astype('float')

# Converting 'is_safe' column datatype to int
df['is_safe'] = df['is_safe'].replace('#NUM!', pd.np.nan)
df['is_safe'] = df['is_safe'].astype('float')
df['is_safe'].dtype

# Checking the correlation between columns
df.corr()

# Checking the count of duplicated rows
df.duplicated().sum()

# Filling missing values with 0 in 'ammonia' and 'is_safe' columns
df['ammonia'] = df['ammonia'].fillna(0)
df['is_safe'] = df['is_safe'].fillna(0)

# Data Analysis
sns.countplot('is_safe', data=df)
plt.show()

sns.lineplot('is_safe', 'bacteria', data=df)
plt.show()

# ...more plots and analysis...

# Splitting the dataset into input (X) and target (y) variables
X = df.drop(['is_safe'], axis=1)
y = df['is_safe']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_predict = lr.predict(X_test)
score = accuracy_score(y_test, y_predict)
score

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_predict = lin_reg.predict(X_test)
lin_mse = mean_squared_error(y_test, lin_predict)
lin_mae = mean_absolute_error(y_test, lin_predict)
lin_mse, lin_mae

