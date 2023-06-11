import pandas as pd
import numpy as np

# Read the CSV file into a DataFrame
df = pd.read_csv('/kaggle/input/blood-transfusion-dataset/transfusion.csv')

# Drop duplicates
df = df.drop_duplicates()

# Check for missing values
df.isnull().sum()

# Fill missing values with appropriate methods
df['column_name'] = df['column_name'].fillna(value)  # Replace 'column_name' and 'value' with actual column name and value

# Convert data types if needed
df['column_name'] = df['column_name'].astype(new_data_type)  # Replace 'column_name' and 'new_data_type' with actual column name and data type

# Perform other preprocessing steps (e.g., feature scaling, encoding categorical variables, etc.)

# Save the preprocessed DataFrame to a new CSV file
df.to_csv('preprocessed_data.csv', index=False)

