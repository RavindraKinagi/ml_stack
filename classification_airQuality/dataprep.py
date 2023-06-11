import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/adl-classification/dataset.csv', names=['MQ1', 'MQ2', 'MQ3', 'MQ4', 'MQ5', 'MQ6', 'CO2'])

def preprocessing(df):
    df = df.copy()
    
    y = df['CO2']
    X = df.drop('CO2', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocessing(data)

