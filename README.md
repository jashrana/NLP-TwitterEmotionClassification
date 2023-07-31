# NLP TwitterEmotionClassification
 We will explore Twitter Emotion Classification. The goal is the identify the primary emotion expressed in a tweet.

# University of Galway
## Assignment 1 - Advanced NLP
For this assignment, we will explore Twitter Emotion Classification. The goal is the identify the primary emotion expressed in a tweet. Consider the following tweets:
```
Tweet 1: @NationalGallery @ThePoldarkian I have always loved this painting.
Tweet 2: '@tateliverpool #BobandRoberta: I am angry more artists that have a profile are not speaking up #foundationcourses.'
``` 

How would you describe the emotions in `Tweet 1` vs `Tweet 2`? `Tweet 1` expresses enjoyment and happiness, while `Tweet 2` directly expresses anger. For this assignment, we will be working with the SMILE Twitter Emotion Dataset ([Wang et al. 2016](https://ceur-ws.org/Vol-1619/paper3.pdf)). At a high level, our goal is to develop different models (rule-based, machine learning, and deep learning), which can be used to identify the emotion of a tweet. You will be required to clean and preprocess the data, generate features for classification, train various models, and evaluate the models. 

## Task 1. Data Cleaning, Preprocessing, and splitting [15 points]
The `data` environment contains the SMILE dataset loaded into a pandas dataframe object. Our dataset has three columns: id, tweet, and label. The `tweet` column contains the raw scraped tweet and the `label` column contains the annotated emotion category. Each tweet is labelled with one of the following emotion labels:
- 'nocode', 'not-relevant' 
- 'happy', 'happy|surprise', 'happy|sad'
- 'angry', 'disgust|angry', 'disgust' 
- 'sad', 'sad|disgust', 'sad|disgust|angry' 
- 'surprise'

### Task 1a. Label Consolidation [ 3 points]
As we can see above the annotated categories are complex. Several tweets express complex emotions like (e.g. 'happy|sad') or multiple emotions (e.g. 'sad|disgust|angry'). The first things we need to do is clean up our dataset by removing complex examples and consolidating others so that we have a clean set of emotions to predict. 

For Task 1a., write code which does the following:
1. Drops all rows which have the label "happy|sad", "happy|surprise", 'sad|disgust|angry', and 'sad|angry'.
2. Re-label 'nocode' and 'not-relevant' as 'no-emotion'.
3. Re-label 'disgust|angry' and 'disgust' as 'angry'.
4. Re-label 'sad|disgust' as 'sad'.

Your updated `data' dataframe should have 3,062 rows and 5 label categories (no-emotion, happy, angry, sad, and surprise).

### Task 1b. Tweet Cleaning and Processing [10 points]
Raw tweets are noisy. Consider the example below: 
```
'@tateliverpool #BobandRoberta: I am angry more artists that have a profile are not speaking up #foundationcourses. 😠'
```
The mention @tateliverpool and hashtag #BobandRoberta are extra noise that don't directly help with understanding the emotion of the text. The accompanying emoji can be useful but needs to be decoded to it text form :angry: first. 

For this task you will fill complete the `preprocess_tweet` function below with the following preprocessing steps:
1. Lower case all text
2. De-emoji the text
3. Remove all hashtags, mentions, and urls
4. Remove all non-alphabet characters except the followng punctuations: period, exclamation mark, and question mark

Hints: 
- For step 2 (de-emoji), consider using the python [emoji](https://carpedm20.github.io/emoji/docs/) library. The `emoji.demojize` method will convert all emojis to plain text. The `emoji` library is installed in cell [52].
- Follow the processing steps in order. For example calling nltk's word_tokenize before removing hashtags and mentions will end up creating seperate tokens for @ and # and cause problems.

To get full credit for this task, the Test 1b must pass. Only modify the  cell containing the `preprocess_tweet` function and do not alter the testing code block. 

After you are satisfied with your code, run the tests. code to ensure your function works as expected. This cell will also create a new column called `cleaned_tweet` and apply the `preproces_tweet` function to all the examples in the dataset. 

### Task 1c. Generating Evaluation Splits [2 points]
Finally, we need to split our data into a train, validation, and test set. We will split the data using a 60-20-20 split, where 60% of our data is used for training, 20% for validation, and 20% for testing. As the dataset is heaviliy imbalanced, make sure you stratify the dataset to ensure that the label distributions across the three splits are roughly equal. 

Store your splits in the variables `train`, `val`, and `test` respectively. 

Hints:
- Use the [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function for this task. You'll have to call it twice to get the validation split. 
- Set the random state so the sampling can be reproduced (we use 2023 for our random state)
- Use the `stratify` parameter to ensure representative label distributions across the splits. 

## Task 2: Naive Baseline Using a Rule-based Classifier [10 points]

Now that we have a dataset, let's work on developing some solutions for emotion classification. We'll start with implementing a simple rule-based classifier which will also serve as our naive baseline. Emotive language (e.g. awesome, feel great, super happy) can be a strong signal as to the overall emotion being by the tweet. For each emotion in our label space (happy, surprised, sad, angry) we will generate a set of words and phrases that are often associated with that emotion. At classification time, the classifier will calculate a score based on the overlap between the words in the tweet and the emotive words and phrases for each of the emotions. The emotion label with the highest overlap will be selected as the prediction and if there is no match the "no-emotion" label will be predicted. We can break the implementation of this rules-based classifier into three steps:
1. Emotive language extraction from train examples 
2. Developing a scoring algorithm
3. Building the end-to-end classification flow 

### Task 2a. Emotive Language Extraction [4 points] 
For this task you will generate a set of unigrams and bigrams that will be used to predict each of the labels. Using the training data you will need to extract all the unique unigrams and bigrams associated with each label (excluding no-emotion). Then you should ensure that the extracted terms for each emotion label do not appear in the other lists. In the real world, you would then manually curate the generated lists to ensure that associated words were useful and emotive. For the assignment, you won't be required to further curate the generated lists.

Once you've identified the appropiate terms, save them as lists stored in the following environment variables: `happy_words`, `surprised_words`, `sad_words`,and `angry_words`. To get full credit for this section, ensure all 2a Tests pass. 

Hints
- We suggest you use Python's [set methods](https://realpython.com/python-sets/) for this task.
- NLTK has a function for extracting [ngrams](https://www.nltk.org/api/nltk.util.html?highlight=ngrams#nltk.util.ngrams). This function expects a list of tokens as input and will output tuples which you'll need to reconvert into strings. 

### Task 2b. Scoring using set overlaps [2 points]

Next we will implement to scoring algorithm. Our score will simply be the count of overlapping terms between tweet text and emotive terms. For this task, finish implementing the code below. To get full credit, ensure Test 2b. is successful. 

### 2c. Rule-based classification [4 points] 
Let put together our rules-based classfication system. Fill out the logic in the `simple_clf`. Given a tweet, `simple_clf` will generate the overlap score
for each of emotion labels and return the emotion label with the highest score. If there is no match amongst the emotions, the classifier will return 'no-emotion'.

To get full credit for this section, your average F1 score most be greater than 0.

## Task 3. Machine learning w/ grammar augmented features [10 points]

Now that we have a naive baseline, let's build a more sophisticated solution using machine learning. Up to this point, we have only considered the words in the tweet as our primary features. The rules-based approach is a very simple bag-of-words classifier. Can we improve performance if we provide some additional linguistic knowledge?

For Task 3 you will do the following:
- Generate part-of-speech features our tweets
- Train two different machine learning classifiers, one with linguistic features and one without
- Evaluate the trained models on the test set

### Task 3a. Grammar Augmented Feature Generation [3 points]
For this task, we will be generating part-of-speech tags for each token in our tweet. Additionally we'll lemmatize the text as well. We will directly include the POS information by appending the tag to the lemma of word itself. For example:
```
Raw Tweet: I am very angry with the increased prices.
POS Augmented Tweet: I-PRP be-VBP very-RB angry-JJ with-IN the-DT increase-VBN price-NNS .-.
```

Complete the `generate_pos_features` using the Spacy library. Once you have an implementation that works, we'll update the `train` and `test` dataframes with a new column called `tweet_with_pos` which contains the output of the `generate_pos_features` method.

### Task 3b. Model Training [5 points]
Next we will train two seperate RandomForest Classifier models. For this task you will generate two sets of input features using the `TfidfVectorizer`. We generate Tfidf statistic on the`cleaned_tweet` and the `tweet_with_pos` columns. 

Once you've generated your features, train two different Random Forest classifiers with the generated features and generate the predictions on the test set for each classifier.

### Task 3c. [2 points]
Generate classification reports for both models. Print the reports below. In a few sentences (no more than 100 words) explain which features were the most effective and why you think that's the case?

## Task 4. Transfer Learning with DistilBERT [10 points]

For this task you will finetune a pretrained language model (DistilBERT) using the huggingface `transformers` library. For this task you will need to:
- Encode the tweets using the BERT tokenizer
- Create pytorch datasets for for the train, val and test datasets
- Finetune the distilbert model for 5 epochs
- Extract predictions from the model's output logits and convert them into the emotion labels.
- Generate a classification report on the predictions.

Ensure you are running the notebook in Google Colab with the gpu runtime enabled for this section.

## Task 5. Model Recommendation [5 points]
In a paragraph (no more than 250 words) answer the following questions:
1. Which of the implemented models would you recommend and why?
After seeing all the three models fail on the labels "sad" and "surprise" due to its lack of training data, I'd say it would go for BERT Transformer route as it still gives us better performance score compared to other two, due to the reason that it provides good weights and bias to the internal nodes on which it determines the result, as compared to RandomForestClassifier and the Rule-based model which has fixed set of defined functions and no learning or shifting of the parameters is taking place. It takes the least time out of the three but performs with very underwhelming results out of the three.


2. Compare the metrics for each models implemted (Rules-Based, Machine Learning w/ POS features, and DistilBERT). What are the pros and con for each model (consider performance both macro performance and label specifc metrics and the computational requirements).<br>
* Rule-Based model implemented has a fixed set of words which shows us a bag of words implementation of an NLP task, which can be good if we have a small program allocated for a smaller scale use, but lacks where a word can mean different things in correlation to others or the sentence itself. The model gives us an accuracy of 48% which is very average considering the training and testing size.

* Machine Learning w/ POS features using RandomForestClassifier is still an ideal choice with accuracy around 77% and w/o POS accuracy of 81% which still gives a good ideology of the fact that we can still use them to opt, and can change the hyperparameters or can even do better preprocessing work or use a good lexical database to support our database for preprocessing, to crux the results to even better scale, while it takes considerably less time than the Transformers and comes closer to it.

* Transformers use GPU to process the data, train and run them with a considerably high accuracy out of the tree at a whopping 87% which is a good performance score in a broad subject like NLP, but requires a good knowledge of the subject and even a bigger corpus of data to transform and encode the data. It becomes good as the data is transformed into numbers because a system works well if numbers are provided to it for processing, and that's what is achieved here. The weights and biases for each node in the layers are well set according to the inputs provided and it is a big reason for this method to work the best, although a lot is required to set metrics using PyTorch.