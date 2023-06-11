from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# Handle missing values
crime = crime.dropna()

# Encode categorical variables
encoder = LabelEncoder()
crime['Category'] = encoder.fit_transform(crime['Category'])
crime['DayOfWeek'] = encoder.fit_transform(crime['DayOfWeek'])
crime['PdDistrict'] = encoder.fit_transform(crime['PdDistrict'])
crime['Resolution'] = encoder.fit_transform(crime['Resolution'])

# Normalize numeric features
scaler = MinMaxScaler()
crime['Dates'] = scaler.fit_transform(crime['Dates'].values.reshape(-1, 1))

# Split the data into features and target variable
X = crime.drop('Category', axis=1)
y = crime['Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

