# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt


stop_words = text.ENGLISH_STOP_WORDS

categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
twenty_2classes = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)


def Counter(min_df, train):
    print("min_df =",min_df)
    count_vect = CountVectorizer(min_df=min_df, stop_words=stop_words)
    X_train_counts = count_vect.fit_transform(train.data)
    XX_train  = X_train_counts.toarray()
    print("Shape of train =",XX_train.shape)
    return X_train_counts

X_counts_3 = Counter(3, twenty_2classes)



def tfidf(X_train_counts):
    tfidf_transformer = TfidfTransformer()
    print(X_train_counts.shape)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    return X_train_tfidf

X_tfidf_3 = tfidf(X_counts_3)
print ("X_tfidf_3.shape =",X_tfidf_3.shape)






