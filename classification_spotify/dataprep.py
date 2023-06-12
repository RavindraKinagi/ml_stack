import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report

# Read the file and do some preprocessing
df = pd.read_csv('/kaggle/input/best-songs-on-spotify-for-every-year-2000-2023/Best Songs on Spotify from 2000-2023.csv', delimiter=';')
dft = pd.read_csv('/kaggle/input/best-songs-on-spotify-for-every-year-2000-2023/Best Songs on Spotify from 2000-2023.csv', delimiter=';')

# Categorize the bins for the classification task
dft["popularity"] = pd.cut(dft["popularity"], bins=[0,50,75,100], labels=[0,1,2], include_lowest=True)
dft["popularity"].value_counts(dropna=False, normalize=True) * 100

# One hot encode method for the artist and the genre
def onehot_encode(df, column, prefix):
    data = df.copy()
    dummies = pd.get_dummies(data[column], prefix=prefix)
    data = pd.concat([data, dummies], axis=1)
    data = data.drop(column, axis=1)
    return data

df = onehot_encode(df, 'artist', 'a')
df = onehot_encode(df, 'top genre', 'g')

# Formatting the new dataset
df = df.drop(['title', 'popularity'], axis=1)
dft = dft.drop(['title','artist','top genre','year','bpm','energy', 'dB','liveness','valence', 'acousticness'], axis=1)
dft = dft.drop(['danceability ','duration','speechiness '], axis=1)

# Save the target dataset to a CSV file
dft.to_csv('target_csv.csv', sep=',', index=False, encoding='utf-8')

# Convert the datasets to NumPy arrays
song_numpy = np.loadtxt('new_csv.csv', dtype=np.float32, delimiter=",", skiprows=1)
target_numpy = np.loadtxt('target_csv.csv', dtype=np.float32, delimiter=",", skiprows=1)

# Convert NumPy arrays to PyTorch tensors
songs_py = torch.from_numpy(song_numpy)
target_py = torch.from_numpy(target_numpy)

# Define the dataset class
class CTDataset(Dataset):
    def __init__(self, songs, popularity):
        self.x = songs
        self.y = popularity
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]

# Create the entire dataset
dataset = CTDataset(songs_py, target_py)

# Define the split ratios
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

# Compute the sizes of each split
num_samples = len(dataset)
train_size = int(train_ratio * num_samples)
val_size = int(val_ratio * num_samples)
test_size = num_samples - train_size - val_size

# Use random_split to split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size

