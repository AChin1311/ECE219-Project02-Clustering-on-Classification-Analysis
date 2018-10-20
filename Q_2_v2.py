
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

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


Class_1 = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
Class_2 = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

truth_Y = []
zeros, ones = 0, 0
for i in range(len(twenty_2classes.target)):
    if twenty_2classes.target_names[twenty_2classes.target[i]] in Class_1:
        truth_Y.append(0)
        zeros += 1
    if twenty_2classes.target_names[twenty_2classes.target[i]] in Class_2:
        truth_Y.append(1)
        ones += 1
print("# of Class 1 = ", zeros)
print("# of Class 2 = ", ones)
print(truth_Y[:10],twenty_2classes.target[:10],[twenty_2classes.target_names[i] for i in twenty_2classes.target[:10]])


def KMeans(X):
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    predict_Y = kmeans.predict(X)
    return predict_Y

predict_Y = KMeans(X_tfidf_3)
print(predict_Y[:30])



def contingency_matrix(truth_Y, predict_Y):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(truth_Y, predict_Y)

print("contingency_matrix = \n", contingency_matrix(truth_Y, predict_Y))



def homogeneity(truth_Y, predict_Y):
    from sklearn.metrics.cluster import homogeneity_score
    return homogeneity_score(truth_Y, predict_Y)

def completeness(truth_Y, predict_Y):
    from sklearn.metrics.cluster import completeness_score
    return completeness_score(truth_Y, predict_Y)

def V_measure(truth_Y, predict_Y):
    from sklearn.metrics.cluster import v_measure_score
    return v_measure_score(truth_Y, predict_Y)

def adjusted_rand(truth_Y, predict_Y):
    from sklearn.metrics.cluster import adjusted_rand_score
    return adjusted_rand_score(truth_Y, predict_Y)

def adjusted_mutual_info(truth_Y, predict_Y):
    from sklearn.metrics.cluster import adjusted_mutual_info_score
    return adjusted_mutual_info_score(truth_Y, predict_Y)

print("homogeneity_score = ", homogeneity(truth_Y, predict_Y))
print("completeness_score = ", completeness(truth_Y, predict_Y))
print("v_measure_score = ", V_measure(truth_Y, predict_Y))
print("adjusted_rand_score = ", adjusted_rand(truth_Y, predict_Y))
print("adjusted_mutual_info_score = ", adjusted_mutual_info(truth_Y, predict_Y))

