# toxic-comment-classification

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



