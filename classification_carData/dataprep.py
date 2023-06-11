import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv("/kaggle/input/cars-purchase-decision-dataset/car_data.csv")

# Drop unnecessary columns
df = df.drop('User ID', axis='columns')

# Perform EDA
df.info()
df.duplicated().sum()
df.isna().sum()
df['Gender'].value_counts()
df['Purchased'].value_counts()

# Visualize data
sns.histplot(x="AnnualSalary", data=df, hue="Gender")
sns.histplot(x="Age", data=df, hue="Gender")
sns.histplot(x="AnnualSalary", data=df, hue="Purchased")
sns.histplot(x="Age", data=df, hue="Purchased")

# Perform feature engineering
df = pd.get_dummies(df, drop_first=False)
df = df.drop("Gender_Male", axis='columns')

# Split the dataset into features and target
X = df[['AnnualSalary', 'Age', 'Gender_Female']].copy()
y = df[['Purchased']].copy()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Scale the features using StandardScaler
scaler = StandardScaler()
Scaled_X_train = scaler.fit_transform(X_train)
Scaled_X_test = scaler.transform(X_test)

