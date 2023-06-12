import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.tokenize import word_tokenize
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Reading Data
data = pd.read_csv('../input/youtube-videos-dataset/youtube.csv')
data.head()
print(data.description[0])
data.shape

# Checking for null values
data.isnull().sum()

# Getting unique values of "category" column
data['category'].unique()

# Getting the frequency count for each category in "category" column
category_valueCounts = data['category'].value_counts()
category_valueCounts

plt.figure(figsize=(12, 8))
plt.pie(category_valueCounts, labels=category_valueCounts.index, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.show()

def showWordCloud(categoryName, notInclude=['subscribers', 'SUBSCRIBE', 'subscribers'], w=15, h=15):
    global data
    print(f"Word Cloud for {categoryName}")
    plt.figure(figsize=(w, h))
    text = " ".join(word for word in data[data.category==categoryName].description.astype(str))
    for word in notInclude:
        text = text.replace(word, "")
    wordcloud = WordCloud(background_color='white', stopwords=STOPWORDS, max_words=90).generate(text)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    
showWordCloud('food', ['subscribers', 'SUBSCRIBE', 'subscribers', 'SHOW', 'video'])
showWordCloud('travel', ['subscribers', 'SUBSCRIBE', 'subscribers', 'SHOW', 'video'])
showWordCloud('history', ['subscribers', 'SUBSCRIBE', 'subscribers', 'SHOW', 'video'])
showWordCloud('art_music', ['subscribers', 'SUBSCRIBE', 'subscribers', 'SHOW', 'video'])

# Data Processing

# 1) Remove Punctuation
punctuation = string.punctuation

def removePunctuationFromText(text):
    text = ''.join([char for char in text if char not in punctuation])
    return text

data['descriptionNonePunct'] = data['description'].apply(removePunctuationFromText)
data.head()

# 2) Tokenize words
data['descriptionTokenized'] = data['descriptionNonePunct'].apply(word_tokenize)
data.head()

# 3) Remove stopwords
stopWords = stopwords.words('english')

def removeStopWords(text):
    return [word for word in text if word not in stopWords]

data['descriptionNoneSW'] = data['descriptionTokenized'].apply(removeStopWords)
data.head()

# 4) Text to Sequence
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(data['descriptionNoneSW'])
data['textSequence'] = tokenizer.texts_to_sequences(data['descriptionNoneSW'])
data.head()

# Model Building
input_shape = int(sum(data['textSequence'].apply(lambda x: len(x) / len(data['textSequence']))))
input_shape

from keras.preprocessing.sequence import pad_sequences
X = pad_sequences(data['textSequence'], maxlen=45)
X

y = data['category']
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)
y

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Getting vocabulary length
maxWords = (max(map(max, X))) + 1

model = keras.models.Sequential([
    keras.layers.Embedding(maxWords, 64, input_shape=[input_shape]),
    keras.layers.GRU(32),
    keras.layers.Dense(4, activation='softmax')
])

# Compiling the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2)

pd.DataFrame(history.history).plot(figsize=(12, 8))
plt.grid(True)
# Set vertical range between 0 and 1
plt.gca().set_ylim(0, 1)

# Evaluating the model
model_evaluated = model.evaluate(X_test, y_test)
print(f'Model evaluated Loss is {model_evaluated[0]}')
print(f'Model evaluated accuracy is {model_evaluated[1]}')

# Confusion Matrix and Classification Report
y_pred = (model.predict(X_test).argmax(axis=-1)).tolist()
class_names = y_encoder.classes_

print("Classification report:\n", classification_report(y_test, y_pred, target_names=class_names))

def drawConfusionMatrix(true, preds, normalize=None):
    confusionMatrix = confusion_matrix(true, preds, normalize=normalize)
    confusionMatrix = np.round(confusionMatrix, 2)
    sns.heatmap(confusionMatrix, annot=True, annot_kws={"size": 12}, fmt="g", cbar=False, cmap="viridis")
    plt.show()

print("Confusion matrix:\n")
drawConfusionMatrix(y_test, y_pred)

print("Normalized confusion matrix:\n")
drawConfusionMatrix(y_test, y_pred, "true")

