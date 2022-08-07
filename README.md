# Turkish-twitter-politician-analysis
In this project, I have written a machine learning model that determines whether the tweets about politicians are positive or negative.
Then, I used the model to predict which politicians are liked more by Turkish people. There are some interesting data, so I recommend you to at least check out the graphs.
You can reach out the dataset I have used [here](https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset). I have filtered some to use in this project.
## Preprocessing Our Data
I will be using the [Zemberek](https://github.com/ahmetaa/zemberek-nlp) library to process our data. Zemberek is a natural language processing (NLP) tool for Turkish language.
The steps that I will be using are:
* Sentence normalization
* Removing punctuations, mentions, digits and emojis
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

## Visualizing the Data
I will be using matplotlib library in python to visualize our data.

### Negative - Positive Balance
![alt text](https://i.imgur.com/m6tOtxw.png)
We can see from the graph that we have a nice balance in our data.

### Most Used Words
![alt text](https://i.imgur.com/nd37R4F.png)
Seems pretty reasonable. Nothing interesting here.

#### Most Used Word Combinations (Bigram)
![alt text](https://i.imgur.com/fhsPYEA.png)
Now that is interesting. Most used 2-word combinations are "orospu çocuk" and "am koy". Since we lemmatized the data, it shows the lemmatized version of very common bad words.
What if don't lemaatize?

![alt text](https://i.imgur.com/UHJDkIl.png)
All right. That is pretty dark. I guess we can conclude that Turkish Twitter community is pretty toxic.

#### Most Used Word Combinations (Trigram)
![alt text](https://i.imgur.com/GxBpnPY.png)
Event though I can pretty much predict the results now, I wanted the see the trigram graph too. At this point, not really interesting.


## Building Our Model

