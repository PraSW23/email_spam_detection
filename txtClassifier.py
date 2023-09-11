import csv
from textblob import TextBlob
import pandas as pd
import sklearn
import cPickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


def split_into_tokens(message):
    message = unicode(message, 'utf8')
    return TextBlob(message).words


def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]


messages = pd.read_csv('./smsspamcollection/SMSSpamCollection', sep='\t', quoting=csv.QUOTE_NONE,
                       names=["label", "message"])


vectorizer = CountVectorizer(analyzer=split_into_lemmas)
messages_Vect = vectorizer.fit_transform(messages.message)


tfidf_transformer = TfidfTransformer()
messages_tfidf = tfidf_transformer.fit_transform(messages_Vect)


spam_detector = MultinomialNB().fit(messages_tfidf, messages['label'])

pipeline = Pipeline([
    ('vec', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)

scores = cross_val_score(pipeline,  # steps to convert raw messages into models
                         msg_train,  # training data
                         label_train,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=-1,  # -1 = use all cores = faster
                         )
# print scores

params = {
    'tfidf__use_idf': (True, False),
}

grid = GridSearchCV(
    pipeline,  # pipeline from above
    params,  # parameters to tune via cross validation
    refit=True,  # fit using all available data at the end, on the best found param combination
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    # cv=StratifiedKFold(label_train),  # what type of cross validation to use
)

nb_detector = grid.fit(msg_train, label_train)
# print nb_detector.grid_scores_

print(nb_detector.predict(["Hi How are you"])[0])

with open('sms_spam_detector.pkl', 'wb') as fout:
    cPickle.dump(nb_detector, fout)
