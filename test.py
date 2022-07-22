

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
from sklearn import metrics


def recall_a(output, labels):
    preds = torch.argmax(output, dim=1)
    labels = labels.to(torch.device("cpu")).numpy()
    preds = preds.to(torch.device("cpu")).numpy()
    macro = metrics.recall_score(labels, preds, average=None)
    return macro


def precision(y_true, y_pred):
    i = set(y_true).intersection(y_pred)
    len1 = len(y_pred)
    if len1 == 0:
        return 0
    else:
        return len(i) / len1


def recall(y_true, y_pred):
    y_true = y_true.to(torch.device("cpu")).tolist()
    y_pred = y_pred.to(torch.device("cpu")).tolist()
    i = set(y_true).intersection(y_pred)
    return len(i) / len(y_true)


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0
    else:
        return 2 * (p * r) / (p + r)

def data_cleaning(text):
    # seems that Uppercase words have more effect on toxicity than lowercase.
    # so I decided to keep them as they are.
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    text = text.replace("#", " ")

    text = re.sub('https?://[A-Za-z0-9./]+', '', text)
    text = re.sub('http?://[A-Za-z0-9./]+', '', text)
    text = re.sub('www.[A-Za-z0-9./]+', '', text)
    encoded_string = text.encode("ascii", "ignore")
    decode_string = encoded_string.decode()
    return decode_string


data_folder_path = './data/'

test_data = pd.read_csv(data_folder_path + 'test.csv')
test_labels = pd.read_csv(data_folder_path + 'test_labels.csv')
df = pd.concat([test_data, test_labels], axis=1)

test_len = sum([label != -1 for label in df['toxic']])


df['clean_comment'] = df['comment_text'].apply(data_cleaning)
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
MAX_LEN = 436
classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

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
                                                  max_length=MAX_LEN,
                                                  padding='max_length',
                                                  truncation=True,
                                                  return_attention_mask=True)
        ids = torch.tensor(tokenized_comment['input_ids'], dtype=torch.long)
        mask = torch.tensor(tokenized_comment['attention_mask'], dtype=torch.long)

        labels = self.labels[idx]
        labels = torch.tensor(labels, dtype=torch.float)
        return {'ids': ids, 'mask': mask, 'labels': labels}


dataset_test = BertDataSet(df)

test_batch = 1

data_loader_test = DataLoader(dataset_test, batch_size=test_batch, shuffle=False, pin_memory=True)
model = transformers.BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=6)
gpus = torch.cuda.device_count()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if gpus > 1:
    print("Let's use", gpus, "GPUs!")
    model = torch.nn.DataParallel(model)  # multi-gpu
model.to(device)
loss = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor((159571 - 35098) / 35098))
loss.to(device)
epochs = 5
LR = 2e-5  # Learning rate
optimizer = torch.optim.AdamW(model.parameters(), LR, weight_decay=1e-2)
torch.backends.cudnn.benchmark = True

for i in range(5):
    model.load_state_dict(torch.load(f'./model_save/{i}_aug_False_loss_False.pkl'))

    labels_total = None
    predictions_total = None
    label_counter = 0
    pred_counter = 0
    with torch.no_grad():
        correct_predictions = 0
        test_losses = []
        for batch_id, batch in enumerate(data_loader_test):
            if batch['labels'][0][0] == -1:
                #print(batch['labels'][0][0])
                continue
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            outputs = model(ids, mask)
            outputs = outputs['logits'].squeeze(-1).to(torch.float32)
            probabilities = torch.sigmoid(outputs)
            predictions = torch.where(probabilities > 0.5, 1, 0)
            labels = batch['labels'].to(device, non_blocking=True)
            loss_valid = loss(outputs, labels)
            # print("prediction shape: ", predictions)
            # print("labels shape: ", labels.to(torch.int))
            # recall_score = recall(labels.to(torch.int), predictions.to(torch.int))
            # print("recall_score", recall_score)
            test_losses.append(loss_valid.item())
            correct_predictions += torch.sum(predictions == labels)

            labels = labels.to(torch.device("cpu")).numpy()
            predictions = predictions.to(torch.device("cpu")).numpy()
            for row in range(len(labels)):
                for col in range(len(labels[row])):
                    if labels[row][col] == 1:
                        label_counter += 1
                        if predictions[row][col] == 1:
                            pred_counter += 1
            '''
            if labels_total == None:
                labels_total = labels
            else:
                labels_total = torch.cat((labels_total, labels), 0)
            if predictions_total == None:
                predictions_total = predictions
            else:
                predictions_total = torch.cat((predictions_total, predictions), 0)
            '''

        accuracy = correct_predictions / (test_len * 6)
        recall_score = pred_counter / label_counter
        # recall_score = recall(labels_total.to(torch.int), predictions_total.to(torch.int))
        print("epoch: {}, recall_score: {}".format(i, recall_score))
        # f1_score = f1(labels_total.to(torch.int), predictions_total.to(torch.int))
        # print("f1", f1_score)
        print('Validation Accuracy: {}, loss: {}'.format(accuracy, np.mean(test_losses)))

