import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Read the CSV file into a DataFrame
df = pd.read_csv('/kaggle/input/bmidataset/bmi.csv')

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert gender to numeric values
df['Gender'] = np.where(df['Gender'] == 'Male', 1, 0)

# Perform SMOTE for class balancing
X = df.drop('Index', axis=1)
Y = df['Index']
smote = SMOTE()
X_resampled, Y_resampled = smote.fit_resample(X, Y)
df = pd.DataFrame(X_resampled, columns=X.columns)
df['Index'] = Y_resampled

# Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('Index', axis=1))
df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_scaled['Index'] = df['Index']

# Save the preprocessed DataFrame to a new CSV file
df_scaled.to_csv('preprocessed_data.csv', index=False)

