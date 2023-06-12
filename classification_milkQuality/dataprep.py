import numpy as np
import pandas as pd

# Read dataset
df = pd.read_csv("/kaggle/input/milkquality/milknew.csv")
display(df.head(2))

# Check basic information about the data
df.info()

# Check unique values in each column
for i in df.columns:
    print(f"Column {i}")
    print(df[i].unique())
    print(f"Unique number of elements in column {i} is {df[i].nunique()}")
    print("")

# Encode the "Grade" column
df.Grade = df.Grade.replace({"high": 3, "medium": 2, "low": 1}).astype("int")
df.head(2)

# Check the spread of the data
pd.DataFrame(df.describe().T)

# Visualize the data
import plotly.express as px
import seaborn as sns

# Correlation matrix
px.imshow(df.corr(), text_auto=True)

# Box plots
px.box(df, template="plotly_white")

# Scatter plot: pH vs. Temperature
px.scatter(df, x="pH", y="Temperature", size="pH", color="pH", template="plotly_white")

# Scatter plot: pH vs. Colour
fig = px.scatter(df, x="pH", y="Colour", color="Colour", size="Colour", template="plotly_white", symbol="Grade")
fig.update_coloraxes(showscale=False)
fig.show()

# Standardize numeric features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric_cols = ["pH", "Temperature", "Colour"]
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Split the data into training and testing sets
x = df.iloc[:, 0:8]
y = df.iloc[:, -1]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85)

# Train a logistic regression model
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression()
logit.fit(x_train, y_train)

