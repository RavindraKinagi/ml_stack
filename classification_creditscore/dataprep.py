import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read the data
df = pd.read_csv("/kaggle/input/credit-score-classification-dataset/Credit Score Classification Dataset.csv")

# Simplify education categories
df["Education"] = df["Education"].replace({"High School Diploma": "Undergraduate", "Associate's Degree": "Undergraduate", 
                                           "Bachelor's Degree": "Graduate", "Master's Degree": "Postgraduate", 
                                           "Doctorate": "Postgraduate"})

# Encode categorical features
label_encoder = LabelEncoder()
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Perform any additional preprocessing steps based on the findings

# Drop unnecessary columns
df.drop(['ID'], axis=1, inplace=True)

# Scale numerical features if necessary

# Split the data into features and target
X = df.drop('Credit Score', axis=1)
y = df['Credit Score']

