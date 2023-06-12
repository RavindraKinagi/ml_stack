import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf

# Load the dataset
df = pd.read_csv('/kaggle/input/gender-classification-dataset/gender_classification_v7.csv')

# Convert 'gender' column to numeric using label encoding
label_encoder = preprocessing.LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])

# Convert 'forehead_width_cm' and 'forehead_height_cm' columns to integers
df['forehead_width_cm'] = df['forehead_width_cm'].astype(int)
df['forehead_height_cm'] = df['forehead_height_cm'].astype(int)

# Split the dataset into training and validation sets
train_df = df.sample(frac=0.75, random_state=4)
val_df = df.drop(train_df.index)

# Scale the data to (0,1) range
max_val = train_df.max(axis=0)
min_val = train_df.min(axis=0)
range = max_val - min_val
train_df = (train_df - min_val) / range
val_df = (val_df - min_val) / range

# Separate the features and labels
X_train = train_df.drop('gender', axis=1)
X_val = val_df.drop('gender', axis=1)
y_train = train_df['gender']
y_val = val_df['gender']

# Define the input shape for the model
input_shape = [X_train.shape[1]]

# Create the Neural Network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=input_shape),
    tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer='l1'),
    tf.keras.layers.Dense(units=32),
    tf.keras.layers.Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mae')

# Train the model
losses = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=256, epochs=25)

# Generate predictions and analyze accuracy
predictions = model.predict(X_val.iloc[0:3, :])
actual_values = y_val.iloc[0:3]

# Visualize training vs validation loss
loss_df = pd.DataFrame(losses.history)
loss_df.loc[:, ['loss', 'val_loss']].plot()

