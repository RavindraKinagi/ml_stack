import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('/kaggle/input/energy-efficiency-data-set/energy_efficiency_data.csv')

# Data exploration
df.head()
df.info()
df.describe()
df.shape

# Visualize the correlation matrix
corr = df.corr()
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
plt.title('Correlation Matrix')
plt.show()

# Histograms of Heating_Load and Cooling_Load
plt.figure(figsize=(8, 6))
plt.hist(df['Heating_Load'], bins=20, color='blue', alpha=0.5)
plt.xlabel('Heating Load')
plt.ylabel('Frequency')
plt.title('Histogram of Heating Load')
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(df['Cooling_Load'], bins=20, color='green', alpha=0.5)
plt.xlabel('Cooling Load')
plt.ylabel('Frequency')
plt.title('Histogram of Cooling Load')
plt.show()

# Missing value visualization
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.title('Missing Values in Dataset')
plt.show()

# Split the dataset into input features (X) and target variables (y)
X = df.iloc[:, :-2]
y = df.iloc[:, -2:]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

