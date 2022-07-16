import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import get_linear_schedule_with_warmup
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import os
## Data Loading

data_folder_path = './data/'
df = pd.read_csv(data_folder_path + 'train.csv')
df.head()
classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df[df[classes[1]] == 1].sample(10)

test_data = pd.read_csv(data_folder_path + 'test.csv')
test_labels = pd.read_csv(data_folder_path + 'test_labels.csv')
test_data = pd.concat([test_data, test_labels], axis=1)
total_samples = df.shape[0]
print("total_samples: ", total_samples)
counter = 0
for cls in classes:
    counter += df[cls].sum()
    rate = df[cls].sum() / total_samples
    rate = np.round(rate*100, 3)
    print(cls +' rate: ', rate, "%")
print("counter: ", counter)
## Data cleaning
def data_cleaning(text):
    # seems that Uppercase words have more effect on toxicity than lowercase.
    # so I decided to keep them as they are.
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    text = text.replace("#" , " ")

    text = re.sub('https?://[A-Za-z0-9./]+', '', text)
    text = re.sub('http?://[A-Za-z0-9./]+', '', text)
    text = re.sub('www.[A-Za-z0-9./]+', '', text)
    encoded_string = text.encode("ascii", "ignore")
    decode_string = encoded_string.decode()
    return decode_string
df['clean_comment'] = df['comment_text'].apply(data_cleaning)
test_data['clean_comment'] = test_data['comment_text'].apply(data_cleaning)
## Tokenization
seed = 42
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=seed)

