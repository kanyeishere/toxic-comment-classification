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

from nltk.corpus import wordnet
from nltk.corpus import stopwords

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
for cls in classes:
    rate = df[cls].sum() / total_samples
    rate = np.round(rate*100, 3)
    print(cls +' rate: ', rate, "%")
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
## augmenting data
toxic_df = df[(df['toxic'] == 1) | (df['severe_toxic'] == 1) | (df['obscene'] == 1) | (df['threat'] == 1) | (df['insult'] == 1) | (df['identity_hate'] == 1)]
non_toxic_df = df[(df['toxic'] == 0) & (df['severe_toxic'] == 0) & (df['obscene'] == 0) & (df['threat'] == 0) & (df['insult'] == 0) & (df['identity_hate'] == 0)]

replacement_rate =0.7
##  synonym replacement
aug_toxic_df = toxic_df.copy(True)
for i, row in toxic_df.iterrows():
    comment = row['clean_comment']
    words = comment.split()
    new_comment = ''
    new_words = []
    for word in words:
        if word in stopwords.words('english'):
            new_words.append(word)
            continue

        random_rate = np.random.uniform(0, 1)

        if random_rate < replacement_rate:
            synonyms = []
            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    synonyms.append(l.name())
            if len(synonyms) > 0:
                new_word = synonyms[np.random.randint(0, len(synonyms))]
                new_words.append(new_word)
            else:
                new_words.append(word)


        else:
            new_words.append(word)
    new_comment = ' '.join(new_words)
    new_row = row.copy(True)
    new_row['clean_comment'] = new_comment
    aug_toxic_df = aug_toxic_df.append(new_row, ignore_index=True)


df_no_aug = df.copy(True)
df_aug = pd.concat([aug_toxic_df, non_toxic_df], ignore_index=True)

## Tokenization
seed = 42
train_df = df_no_aug
train_df.head()
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
# encoded_comment = [tokenizer.encode(sent, add_special_tokens=True) for sent in train_df['clean_comment']]

# comment_len = [len(x) for x in encoded_comment]
# np.max(comment_len), np.quantile(comment_len, 0.97), np.mean(comment_len), np.median(comment_len), np.min(comment_len)

MAX_LEN = 436
class BertDataSet(Dataset):
    def __init__(self, dataframe):
        self.comments = dataframe['clean_comment'].values
        self.labels = dataframe[classes].to_numpy()

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = self.comments[idx]
        tokenized_comment = tokenizer.encode_plus(comment,
                                                    add_special_tokens=True,
                                                    max_length = MAX_LEN,
                                                    padding='max_length',
                                                    truncation = True,
                                                    return_attention_mask = True)
        ids = torch.tensor(tokenized_comment['input_ids'], dtype=torch.long)
        mask = torch.tensor(tokenized_comment['attention_mask'], dtype=torch.long)

        labels = self.labels[idx]
        labels = torch.tensor(labels, dtype=torch.float)
        return {'ids': ids, 'mask': mask, 'labels': labels}

dataset_train = BertDataSet(train_df)
# dataset_test = BertDataSet(valid_df)
# len(dataset_train), len(dataset_test)

train_batch = 52
test_batch = 52
data_loader_train = DataLoader(dataset_train, batch_size=train_batch, shuffle=True, pin_memory = True)
# data_loader_test = DataLoader(dataset_test, batch_size=test_batch, shuffle=False, pin_memory = True)
model = transformers.BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels = 6)
gpus = torch.cuda.device_count()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if gpus > 1:
    print("Let's use", gpus, "GPUs!")
    model = torch.nn.DataParallel(model)    # multi-gpu
model.to(device)
# loss = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor((159571 - 35098) / 35098))
loss = torch.nn.BCEWithLogitsLoss()
loss.to(device)

epochs = 5
LR = 2e-5 #Learning rate
optimizer = torch.optim.AdamW(model.parameters(), LR, weight_decay = 1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 2, verbose = True)
torch.backends.cudnn.benchmark = True

for i in range(epochs):
    model.train()
    correct_predictions = 0
    for batch_id, batch in enumerate(data_loader_train):
        optimizer.zero_grad()
        train_losses = []
        with torch.cuda.amp.autocast():
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            optimizer.zero_grad()
            outputs = model(ids, mask)
            outputs = outputs['logits'].squeeze(-1).to(torch.float32)
            probabilities = torch.sigmoid(outputs)
            predictions = torch.where(probabilities > 0.5, 1, 0)
            labels = batch['labels'].to(device, non_blocking=True)
            loss_value = loss(outputs, labels)
            train_losses.append(loss_value.item())
            loss_value.backward()
            correct_predictions += torch.sum(predictions == labels)
        optimizer.step()
        if batch_id % 10 == 0:
            print('Epoch: {}, Batch: {}, Loss: {}'.format(i, batch_id, np.mean(train_losses)))
    accuracy = correct_predictions/(len(dataset_train)*6)
    print('Epoch: {}, Accuracy: {}'.format(i, accuracy))
    model.eval()
    # test
    #with torch.no_grad():
    #    correct_predictions = 0
    #    test_losses = []
    #    for batch_id, batch in enumerate(data_loader_test):
    #        ids = batch['ids'].to(device)
    #        mask = batch['mask'].to(device)
    #        outputs = model(ids, mask)
    #        outputs = outputs['logits'].squeeze(-1).to(torch.float32)
    #        probabilities = torch.sigmoid(outputs)
    #        predictions = torch.where(probabilities > 0.5, 1, 0)
    #        labels = batch['labels'].to(device, non_blocking=True)
    #        loss_valid = loss(outputs, labels)
    #        test_losses.append(loss_valid.item())
    #        correct_predictions += torch.sum(predictions == labels)
    torch.save(model.state_dict(), './model_save/{}_aug_False_loss_False.pkl'.format(i))
    #accuracy = correct_predictions/(len(dataset_test)*6)
    #print('Epoch: {}, Validation Accuracy: {}, loss: {}'.format(i, accuracy, np.mean(test_losses)))
print(torch.cuda.memory_summary(1))


