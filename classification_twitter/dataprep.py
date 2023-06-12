# Importing the dataset
df = pd.read_csv('../input/twitter-tweets-sentiment-dataset/Tweets.csv', header=0)

# Checking for null values
np.sum(df.isnull())

# Dropping rows with null values
df = df.dropna()
df = df.reset_index(drop=True)

# Splitting the dataset into input (X) and target (y)
X = df['selected_text']
y = df['sentiment']

# Encoding the target labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
train_sentences, test_sentences, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

# Printing the shapes of the training and testing sets
print(f'Total training samples: {train_sentences.shape}\n')
print(f'Total training labels: {train_labels.shape}\n')
print(f'Total test samples: {test_sentences.shape}\n')
print(f'Total test labels: {test_labels.shape}\n')

