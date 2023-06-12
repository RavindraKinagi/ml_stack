import time
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from sklearn.utils import resample

df = pd.read_csv("/kaggle/input/emotions-in-text/Emotion_final.csv")
df.head()
sns.countplot(x="Emotion", data=df)

# Upsample surprise
n_surp_sample = 1000
surp = df[df["Emotion"] == "surprise"]
surp_upsample = resample(surp, random_state=35, n_samples=n_surp_sample, replace=True)

# Upsample love
n_love_sample = 500
love = df[df["Emotion"] == "love"]
love_upsample = resample(love, random_state=35, n_samples=n_love_sample, replace=True)

df = pd.concat([df, surp_upsample, love_upsample])
sns.countplot(x="Emotion", data=df)

seq_len = (len(i.split()) for i in df['Text'])
pd.Series(seq_len).hist(bins=50)

encoder = OneHotEncoder()
X = np.array(df.Text)
y = encoder.fit_transform(np.array(df.Emotion).reshape(-1, 1)).toarray()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train)

tokenize_data = Tokenizer(oov_token='<UNK>', split=" ")
tokenize_data.fit_on_texts(X)

tokenize_train = tokenize_data.texts_to_sequences(X_train)
vec_train = pad_sequences(tokenize_train, padding="post", maxlen=50)

tokenize_val = tokenize_data.texts_to_sequences(X_val)
vec_val = pad_sequences(tokenize_val, padding="post", maxlen=50)

tokenize_test = tokenize_data.texts_to_sequences(X_test)
vec_test = pad_sequences(tokenize_test, padding="post", maxlen=50)

vocab_size = len(tokenize_data.word_index) + 1

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional, Dropout, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=50))
model.add(Bidirectional(LSTM(units=256, dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(units=256, dropout=0.2, return_sequences=True)))
model.add(GlobalAveragePooling1D())
model.add(Dense(units=512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(units=6, activation='softmax'))

opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint("/kaggle/working/model.pth", monitor='val_accuracy', verbose=0, save_best_only=True, mode='auto')

t = time.time()
his = model.fit(x=vec_train, y=y_train, batch_size=128, epochs=

