import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read the CSV file
data = pd.read_csv('../input/patient-treatment-classification/data-ori.csv')

# Function to preprocess inputs
def preprocess_inputs(df):
    df = df.copy()
    
    # Binary encoding
    df['SEX'] = df['SEX'].replace({'F': 0, 'M': 1})
    
    # Split df into X and y
    y = df['SOURCE']
    X = df.drop('SOURCE', axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
    
    # Scale X
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    
    return X_train, X_test, y_train, y_test

# Preprocess the inputs
X_train, X_test, y_train, y_test = preprocess_inputs(data)

