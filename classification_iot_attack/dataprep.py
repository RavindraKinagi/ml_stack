import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

def current_milli_time():
    return round(time.time() * 1000)

def calculate_AR(confusion_matrix):
    return (confusion_matrix[0][0] + confusion_matrix[1][1]) / (confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1])

def print_values(algorithm:str, accuracy_rate:float, training_time:int):
    print(algorithm, "Accuracy rate", round(accuracy_rate*100,2), "Training time:", training_time)

# Load the dataset
data = pd.read_csv("/kaggle/input/iot-attack-prediction-dataset/All_Attacks.csv",sep=";")
data.fillna(method='bfill', inplace=True)
data = data.drop_duplicates(inplace=False)

# Feature importance ranking
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

X = data.iloc[:, 0:15]
y = data.iloc[:, -1]

model = ExtraTreesClassifier()
model.fit(X, y)
print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(12).plot(kind='barh')
plt.show()

# Split the data into dependent and independent variables
data_copy = data.copy()
data_copy.drop(['label'], axis=1, inplace=True)

y = data['label']
X = data_copy

# Split the data into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, shuffle=True)

# Decision Tree Classifier
dtstart_time = current_milli_time()
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(x_train, y_train)
dtend_time = current_milli_time()
DTduration = dtend_time - dtstart_time
y_pred_dtc = dtc.predict(x_test)
cm_dtc = confusion_matrix(y_test, y_pred_dtc)
ar_dtc = calculate_AR(cm_dtc)

# Naive Bayes Classifier
nbstart_time = current_milli_time()
gnb = GaussianNB()
gnb.fit(x_train, y_train)
nbend_time = current_milli_time()
NBduration = nbend_time - nbstart_time
y_pred_nb = gnb.predict(x_test)
cm_nb = confusion_matrix(y_test, y_pred_nb)
ar_nb = calculate_AR(cm_nb)

# KNN Classifier
knnstart_time = current_milli_time()
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
knnend_time = current_milli_time()
knnduration = knnend_time - knnstart_time
y_pred_knn = knn.predict(x_test)
cm_knn = confusion_matrix(y_test, y_pred_knn)
ar_knn = calculate_AR(cm_knn)

# Random Forest Classification
rfstart_time = current_milli_time()
rfc = RandomForestClassifier(n_estimators=860, criterion='entropy')
rfc.fit(x_train, y_train)
rfend_time = current_milli_time()
RFduration = rfend_time - rfstart_time
y_pred_rfc = rfc.predict(x_test)
cm_rfc = confusion_matrix(y

