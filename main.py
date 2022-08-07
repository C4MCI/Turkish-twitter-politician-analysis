import os
import string
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from typing import List
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java, isJVMStarted
import pickle

ZEMBEREK_PATH = r'data\zemberek-full.jar'
DATA_PATH = "data"
if isJVMStarted() is False:
    startJVM(getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % ZEMBEREK_PATH)
TurkishMorphology = JClass('zemberek.morphology.TurkishMorphology')
TurkishSentenceNormalizer: JClass = JClass(
    "zemberek.normalization.TurkishSentenceNormalizer"
)
TurkishTokenizer: JClass = JClass("zemberek.tokenization.TurkishTokenizer")
Paths: JClass = JClass("java.nio.file.Paths")
morphology = TurkishMorphology.createWithDefaults()
normalizer = TurkishSentenceNormalizer(
    TurkishMorphology.createWithDefaults(),
    Paths.get(str(os.path.join(DATA_PATH, "normalization"))),
    Paths.get(str(os.path.join(DATA_PATH, "lm", "lm.2gram.slm"))),
)
tokenizer = TurkishTokenizer.DEFAULT


def lemmatize(words):
    if words:
        analysis: java.util.ArrayList = (
            morphology.analyzeAndDisambiguate(words).bestAnalysis()
        )
        pos: List[str] = []
        for i, analysis in enumerate(analysis, start=1):
            f'\nAnalysis {i}: {analysis}',
            f'\nPrimary POS {i}: {analysis.getPos()}'
            f'\nPrimary POS (Short Form) {i}: {analysis.getPos().shortForm}'
            if str(analysis.getLemmas()[0]) != "UNK":
                pos.append(
                    f'{str(analysis.getLemmas()[0])}'
                )
            else:
                pos.append(f'{str(analysis.surfaceForm())}')

        return " ".join(pos)
    else:
        return words


def tokenize(text):
    tokens = []
    for i, token in enumerate(tokenizer.tokenizeToStrings(JString(text))):
        tokens.append(str(token))
    return tokens


def beautify(data):
    stop_words = set(stopwords.words("turkish"))
    data["text"] = data["text"].apply(lambda x: str(normalizer.normalize(JString(x))))
    data["text"] = data["text"].apply(lambda x: "".join([i for i in x if i not in string.punctuation]))
    data["text"] = data["text"].apply(lambda x: "".join([i for i in x if not i.isdigit()]))
    data["text"] = data["text"].apply(lambda x: tokenize(x))
    data["text"] = data["text"].apply(lambda x: [i for i in x if i not in stop_words])
    data["text"] = data["text"].apply(lambda x: [lemmatize(i) for i in x])
    data["text"] = data["text"].apply(lambda x: " ".join([i for i in x]))

    return data


data = pd.read_csv("train_tweet.csv")
data = beautify(data)

tf = TfidfVectorizer()
text_counts = tf.fit_transform(data["text"])
tf_filename = 'finalized_vectorizer_tweet.sav'
pickle.dump(tf, open(tf_filename, "wb"))

test_data = pd.read_csv("test_tweet.csv")
test_data = beautify(test_data)
X_test = tf.transform(test_data["text"])
y_test = test_data["label"]

X_train, y_train = text_counts, data["label"]
le = LabelEncoder()
y_train_xgb = le.fit_transform(y_train)
y_test_xgb = le.fit_transform(y_test)

clf = MultinomialNB()
history = clf.fit(X_train, y_train)

lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)

xgb = XGBClassifier()
xgb.fit(X_train, y_train_xgb)

bayes_model_filename = 'finalized_model_bayes_tweet.sav'
lr_model_filename = 'finalized_model_lr_tweet.sav'
xgb_model_filename = 'finalized_model_xgb_tweet.sav'
pickle.dump(clf, open(bayes_model_filename, "wb"))
pickle.dump(lr, open(lr_model_filename, "wb"))
pickle.dump(xgb, open(xgb_model_filename, "wb"))


predicted_bayes = clf.predict(X_test)
print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted_bayes))

predicted_lr = lr.predict(X_test)
print("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, predicted_lr))


predicted_xgb = xgb.predict(X_test)
print("XGBoost Accuracy:", metrics.accuracy_score(y_test_xgb, predicted_xgb))


