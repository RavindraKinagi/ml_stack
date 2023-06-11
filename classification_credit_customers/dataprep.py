import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Read the data
df = pd.read_csv('/kaggle/input/credit-risk-customers/credit_customers.csv')

# Drop duplicates
df.drop_duplicates(inplace=True)

# Encode categorical features
label_encoder = LabelEncoder()
object_cols = df.select_dtypes(include='object').columns
for col in object_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Feature Engineering
mean_item_amount = df.groupby('purpose')['credit_amount'].mean().to_dict()
df['credit_amount_ratio'] = df.apply(lambda r: 'above mean' if r['credit_amount'] > mean_item_amount[r['purpose']] else 'below mean', axis=1)
df['monthly_return'] = df['credit_amount'] / df['duration']
df['duration_age_ratio'] = 100 * df['duration'] / (12 * df['age'])
df[['sex', 'marriage']] = df['personal_status'].str.split(" ", expand=True)
df['employment'].replace(['unemployed', '<1', '1<=X<4', '4<=X<7', '>=7'], ['unstable', 'unstable', 'stable', 'stable', 'stable'], inplace=True)
df['job'].replace(['unemp/unskilled non res', 'unskilled resident', 'skilled', 'high qualif/self emp/mgmt'], ['unskilled resident', 'unskilled resident', 'skilled', 'high qualif/self emp/mgmt'], inplace=True)
df['housing'].replace(['for free', 'rent', 'own'], ['dont own', 'dont own', 'own'], inplace=True)
df['other_payment_plans'].replace(['none', 'stores', 'bank'], ['non', 'exist', 'exist'], inplace=True)
df['property_magnitude'].replace(['no known property', 'life insurance', 'car', 'real estate'], ['no known property', 'exist', 'exist', 'exist'], inplace=True)

# Drop unnecessary columns
df.drop(['personal_status', 'marriage', 'age', 'duration', 'duration_age_ratio', 'monthly_return', 'credit_amount'], axis=1, inplace=True)

# Scale numerical features
scaler = StandardScaler()
numerical_cols = df.select_dtypes(include='number').columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Split the data into training and testing sets
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE for handling class imbalance
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

