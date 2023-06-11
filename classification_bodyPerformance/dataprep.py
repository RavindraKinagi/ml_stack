import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Load the data
data = pd.read_csv('C:/Users/tomwr/Datascience/Datasets/Tabular/body_performance_multiclass_classification.csv')

# Convert data types
data['age'] = data['age'].astype('int64')
data['diastolic'] = data['diastolic'].astype('int64')
data['systolic'] = data['systolic'].astype('int64')

# Perform feature engineering
data['BMI'] = data['weight_kg'] / (data['height_cm'] / 100) ** 2
data['relative_jump'] = data['broad jump_cm'] / data['height_cm']

# Convert gender to numerical values
data['gender'].replace(['M', 'F'], [0, 1], inplace=True)

# Convert class labels to one-hot encoded format
data = pd.get_dummies(data, columns=['class'])

# Remove unnecessary columns
data.drop(columns=['dummy'], inplace=True)

# Split the dataset into training, validation, and test sets
X = data.iloc[:, :13].values
y = data[['class_A', 'class_B', 'class_C', 'class_D']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

# Scale the features using RobustScaler
scaler = RobustScaler(quantile_range=(15, 85))
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

