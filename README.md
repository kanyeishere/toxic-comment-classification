# toxic-comment-classification

## Data statistics:

<img width="245" alt="image" src="https://user-images.githubusercontent.com/82253442/182348639-77ac8531-2f48-4b64-8c86-5c516ea616d8.png">

![image](https://user-images.githubusercontent.com/82253442/182348812-46d83328-93ec-4e04-aeb5-3da6c017747c.png)

## Data Augmentation
Data augment is necessary for the imbalanced dataset. We used Natural language toolkit performed 50% synonym replacement and stopwords removal on the data.

<img width="445" alt="image" src="https://user-images.githubusercontent.com/82253442/182352067-440f9af8-3101-4ee8-8cfa-b869850ffd61.png">

## Data generation framework:
<img width="494" alt="image" src="https://user-images.githubusercontent.com/82253442/182352301-042f2e05-94bf-48c0-bbb3-1f42e4e10222.png">


## Class-Balanced Loss Weight

![image](https://user-images.githubusercontent.com/82253442/182350591-13a60127-5b28-4e86-8efa-e1241e9a6fe4.png)

To counter the issue of the imbalanced dataset, we modify the weight hyper parameter in the loss function, which gives weight to positive samples for each class. To find a suitable weight, we apply the method proposed by the paper [https://arxiv.org/abs/1901.05555], which is based on the effective number of samples, is used to handle the imbalance. The weights for each class in the loss function are calculated with (1 − β)/(1 − βn) where β = (N − 1)/N (we use 0.9999), and n is the size of samples in the class, N is total number of all classes.


## Experimental Results:
Here, we report our experimental results and we aim to answer the two questions we asked before our experiments:

Question 1: Can our simple model Bi-LSTM beat the BERT pre-trained model in some certain metrics? 

Question 2: Is our Bi-LSTM model lighter than the BERT pre-trained model?

### Overviews:

![image](https://user-images.githubusercontent.com/82253442/182342297-640c973a-7fa9-4337-acb8-ace684d6ea8a.png)

<img width="364" alt="image" src="https://user-images.githubusercontent.com/82253442/182342425-7a6dd241-84e8-41e9-9da8-17ba5c24676e.png">

### Comparisions on different metrcis of muti-classification task:

#### Precision:

<img width="546" alt="image" src="https://user-images.githubusercontent.com/82253442/182343063-8570e665-3279-42e9-bd96-4753e8cacc20.png">

#### Recall:

<img width="554" alt="image" src="https://user-images.githubusercontent.com/82253442/182343242-6064b71c-3f65-4878-aaad-6c9cb2c8538d.png">

#### F1-score:

<img width="555" alt="image" src="https://user-images.githubusercontent.com/82253442/182347002-76284708-7110-4d3f-b1f9-d6541f5a145b.png">

### Analysis:

The comparison of the precision of different models on multi-class classification task.We mark the Bi-LSTM with balanced loss weight and data augmentation as black curve and BERT as blue curve. We can see that in some labels like severe toxic, threat, insult, LSTM with balance weight and data augmentation has higher precision. 

By comparing the recall of different models. We can see the LSTM model improved significantly with the application of data augmentation and weighted loss function. While these techniques make little difference on BERT, so we didn't plot the results of BERT with these techniques on above graphs.

Bert has the highest recall in the majority labels. But the recall of LSTM with balance weight data and data augmentation is very close to the bert implementation, even better on detecting identity hate text.

The accuracy of all above models with data augmentation and balanced loss weight are around 0.97. But because we use imbalanced data, we care F1-score more than accuracy(accuracy is a weak evaluation metric for imbalanced data). And we can see that the overall F1 scores of bert and our lstm are very close, lstm can even beat bert on some certain labels. Which prove that the overall performance of our bi-lstm is neck to neck to BERT.

## Conclusion and Future Work

In conclusion, The self-developed LSTM-based network performs neck to neck with the  BERT pre-trained models with extended tuning and pre-trained word embedding. Which is a surprise considering the current hype that pre-trained models received.

From our experiment, we learnt that the advantages of pre-trained models are

- Easy to use - plenty of APIs, no need to build the deep learning network yourself.
- The hard work of hyperparameters tuning and optimization is done.
- Works great on smaller datasets.
- Works extremely well if your dataset is similar to the dataset that the pre-trained model is trained on

And the disadvantages are 

- Computational intensive to train and deploy.
- Less flexible: Not able to change the network architecture.
- Not suitable to every task.

In our opinion, pre-trained models should only be used as a baseline method or if the dataset is too small and data augmentation and generation are impossible. For large-scale deployment and research, it is almost always better to use a self-developed model than a pre-trained model.

For future work 

- More data augmentation - Rephrase, translate it to another language and back to english.
- Data generation - LSTM GAN etc
- Word embedding - self traiend embedding rather than pre-trained embedding 
- Loss function and Pseudo-Labeling
- Ensemble method




