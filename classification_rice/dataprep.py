import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# Read the dataset
df = pd.read_csv("/kaggle/input/rice-type-classification/riceClassification.csv")
df.head()

# Visualize class distribution for each feature
for label in df.columns[:-1]:
    plt.hist(df[df["Class"]==1][label], color='blue', label='Jasmine', alpha=0.7, density=True)
    plt.hist(df[df["Class"]==0][label], color='red', label='Gonen', alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()

# Split the dataset into train, validation, and test sets
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

# Function to scale the dataset and perform oversampling if specified
def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    data = np.hstack((X, np.reshape(y, (-1, 1))))

    return data, X, y

# Scale and oversample the train, validation, and test sets
train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)

# Train the K-Nearest Neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Predict using the trained model
y_pred = knn_model.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))

