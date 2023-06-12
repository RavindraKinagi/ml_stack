import numpy as np
import pandas as pd

# Read the CSV file
df = pd.read_csv('/kaggle/input/obesity-classification-dataset/Obesity Classification.csv')

# Drop the 'ID' column
df.drop('ID', axis=1, inplace=True)

# Convert the 'Gender' column to numerical values using LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Split the data into input features (x) and target variable (y)
x = df.drop('Label', axis=1)
y = df['Label']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

