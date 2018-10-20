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

def Counter(min_df, train):
    print("min_df =",min_df)
    count_vect = CountVectorizer(min_df=min_df, stop_words=stop_words)
    X_train_counts = count_vect.fit_transform(train.data)
    XX_train  = X_train_counts.toarray()
    print("Shape of train =",XX_train.shape)
    return X_train_counts

def tfidf(X_train_counts):
    tfidf_transformer = TfidfTransformer()
    print(X_train_counts.shape)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    return X_train_tfidf

def contingency_matrix(truth_Y, predict_Y):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(truth_Y, predict_Y)

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


newsgroups20_data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
X_counts20_3 = Counter(3, newsgroups20_data)
X_tfidf20_3 = tfidf(X_counts20_3)
truth20_Y = newsgroups20_data.target

print ("X_tfidf20_3.shape =",X_tfidf20_3.shape)
print("truth20_Y: ",len(truth20_Y))
print(X_tfidf20_3.shape)

def KMeans(X, n):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
    predict_Y = kmeans.predict(X)
    return predict_Y

def SVD(X_train_tfidf, r=1000):
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=r)
    W_train_svd = svd.fit_transform(X_train_tfidf)
    return W_train_svd

print("SVD")
W_svd20_1000 = SVD(X_tfidf20_3)
print("Dim of SVD =",W_svd20_1000.shape)

svd_measures = []
r_values = [1, 2, 3, 5, 10, 20, 50, 100, 300]
for r in r_values:
    retain_W = W_svd20_1000[:,0:r+1]
    predict_Y = KMeans(retain_W, 20)
    svd_measures.append(completeness(truth20_Y, predict_Y))

print(svd_measures, svd_measures.index(max(svd_measures))) 
best_svd_r = r_values[svd_measures.index(max(svd_measures))]
print("SVD max homogeneity r = ", best_svd_r)

def NMF(X_train_tfidf, r):
    from sklearn.decomposition import NMF
    nmf = NMF(n_components=r, init='random', random_state=0)
    W_train_nmf = nmf.fit_transform(X_train_tfidf)
    return W_train_nmf

print("NMF")
best_homo = 0
hest_r = 0
for r in r_values:
    W_nmf20 = NMF(X_tfidf20_3, r)
    print(r, "Dim of NMF =",W_nmf20.shape)
    predict_Y = KMeans(W_nmf20, 20)
    homo = homogeneity(truth20_Y, predict_Y)
    if homo >= best_homo:
        best_homo = homo
        best_nmf_r = r
    print(r, best_homo, best_nmf_r)

print("NMF max homogeneity r = ", best_nmf_r)


# ==============================
# NMF Result
# 1 Dim of NMF = (18846, 1)
# 1 0.0280505908934 1
# 2 Dim of NMF = (18846, 2)
# 2 0.167944732821 2
# 3 Dim of NMF = (18846, 3)
# 3 0.2040192473 3
# 5 Dim of NMF = (18846, 5)
# 5 0.276463589112 5
# 10 Dim of NMF = (18846, 10)
# 10 0.318821507155 10
# 20 Dim of NMF = (18846, 20)
# 20 0.318821507155 10
# 50 Dim of NMF = (18846, 50)
# 50 0.318821507155 10
# 100 Dim of NMF = (18846, 100)
# 100 0.318821507155 10
# 300 Dim of NMF = (18846, 300)
# 300 0.318821507155 10
# NMF max homogeneity r =  10
# ==============================


# best_svd_r = 300
# best_nmf_r = 10

best_svd = W_svd20_1000[:,0:best_svd_r+1]
best_nmf = NMF(X_tfidf20_3, best_nmf_r)
print("done")


def measure(truth_Y, predict_Y):
    print("homogeneity_score = ", homogeneity(truth_Y, predict_Y))
    print("completeness_score = ", completeness(truth_Y, predict_Y))
    print("v_measure_score = ", V_measure(truth_Y, predict_Y))
    print("adjusted_rand_score = ", adjusted_rand(truth_Y, predict_Y))
    print("adjusted_mutual_info_score = ", adjusted_mutual_info(truth_Y, predict_Y))

print("best_nmf_norm")
best_nmf_norm = best_nmf / best_nmf.std(axis = 0)
predict_Y = KMeans(best_nmf_norm, 20)
measure(truth20_Y, predict_Y)

print("best_svd_norm")
best_svd_norm = best_svd / best_svd.std(axis = 0)
predict_Y = KMeans(best_svd_norm, 20)
measure(truth20_Y, predict_Y)

# Logarithm
print("best_nmf_log")
best_nmf_log = np.log( best_nmf + 1e-7 )
predict_Y = KMeans(best_nmf_log, 20)
measure(truth20_Y, predict_Y)

# Logarithm + Norm
print("best_nmf_log_norm")
best_nmf_log_norm = best_nmf_log / best_nmf_log.std(axis = 0)
predict_Y = KMeans(best_nmf_log_norm, 20)
measure(truth20_Y, predict_Y)

# Norm + Logarithm
print("best_nmf_norm_log")
best_nmf_norm_log = np.log(best_nmf_norm+ 1e-7)
predict_Y = KMeans(best_nmf_norm_log, 20)
measure(truth20_Y, predict_Y)

