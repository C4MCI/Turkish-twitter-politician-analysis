In this project, I have written a machine learning model that determines whether the tweets about politicians are positive or negative.
Then, I used the model to predict which politicians are liked more by the Turkish people. There are some interesting data, so I recommend you to at least check out the graphs.
You can reach out to the dataset I have used [here](https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset). I have filtered some and created a new dataset to use in this project.
# Preprocessing Our Data
I will be using the [Zemberek](https://github.com/ahmetaa/zemberek-nlp) library to process our data. Zemberek is a natural language processing (NLP) tool for the Turkish language.
The steps that I will be using are:
* Sentence normalization
* Removing punctuations, mentions, digits, and emojis
* Tokenization
* Removing stop words
* Lemmatization

```python
data["text"] = data["text"].apply(lambda x: str(normalizer.normalize(JString(x))))
data["text"] = data["text"].apply(lambda x: "".join([i for i in x if i not in string.punctuation]))
data["text"] = data["text"].apply(lambda x: "".join([i for i in x if not i.isdigit()]))
data["text"] = data["text"].apply(lambda x: tokenize(x))
data["text"] = data["text"].apply(lambda x: [i for i in x if i not in stop_words])
data["text"] = data["text"].apply(lambda x: [lemmatize(i) for i in x])
data["text"] = data["text"].apply(lambda x: " ".join([i for i in x]))
```

# Visualizing the Data
I will be using the matplotlib library in python to visualize our data.

## Negative - Positive Balance
![alt text](https://i.imgur.com/m6tOtxw.png)
We can see from the graph that we have a nice balance in our data.

## Most Used Words
![alt text](https://i.imgur.com/nd37R4F.png)
Seems pretty reasonable. Nothing interesting here.

### Most Used Word Combinations (Bigram)
![alt text](https://i.imgur.com/fhsPYEA.png)
Now that is interesting. The most used 2-word combinations are "orospu çocuk" and "am koy". Since we lemmatized the data, it shows the lemmatized version of very common bad words.
What if we don't lemmatize?

![alt text](https://i.imgur.com/UHJDkIl.png)
All right. That is pretty dark. I guess we can conclude that the Turkish Twitter community is pretty toxic.

### Most Used Word Combinations (Trigram)
![alt text](https://i.imgur.com/GxBpnPY.png)
Even though I can pretty much predict the results now, I wanted the see the trigram graph too. At this point, not really interesting.


# Building Our Model
So we have an idea about the data we will use, thanks to the visualization. We can start building our machine learning model.

## Vectorizing texts
Now that we processed our text, we need to somehow represent it with numbers to use for our machine learning model. I will be using TF - IDF Vectorizer from sklearn.
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()
text_counts = tf.fit_transform(data["text"])
```

## Choosing the best model
We are trying to build a classification model, so there are some algorithms that we can use. I have picked three algorithms that I believe will yield the best results. These are the Naive Bayes Classifier, Logistic Regression, and XGBoost Classifier.
Let's try them and see which one is working best with our data.

```python
clf = MultinomialNB()
clf.fit(X_train, y_train)

lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)

xgb = XGBClassifier()
xgb.fit(X_train, y_train_xgb)
```

![alt text](https://i.imgur.com/HkWLx0Y.png)

It seems like Logistic regression is working best with our data with a 92.5% accuracy score. I will admit that 92.5% accuracy is not great, but given our dataset is very limited, I will just call it acceptable.
Since we will be using the Logistic Regression model in our analysis, let's save it so we don't have to process all this data again.

```python
pickle.dump(lr, open(lr_model_filename, "wb"))
```

# Analysis About Turkish Politicians
Now that we have a model that can accurately predict whether a tweet is negative or positive, we can fetch some tweets about politicians and use our model to find out public opinion about that politician.
I will be comparing two Turkish politicians. Kemal Kılıçdaroğlu and Mansur Yavaş.
For those who don't know who are these people, they are two of the most possible president candidates from the opposition party.

## Kemal Kılıçdaroğlu



