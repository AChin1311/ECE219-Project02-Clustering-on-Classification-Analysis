#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def SVD(X_tfidf):
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=1000)
    W_train_svd = svd.fit_transform(X_tfidf)
    return W_train_svd




W_svd_3 = SVD(X_tfidf_3)    

print("Dim of SVD =",W_svd_3.shape)


def NMF(X_tfidf):
    from sklearn.decomposition import NMF
    nmf = NMF(n_components=2, init='random', random_state=0)
    W_train_nmf = nmf.fit_transform(X_tfidf)
    return W_train_nmf

W_nmf_3 = NMF(X_tfidf_3)    



print("Dim of NMF =",W_nmf_3.shape)


print("completed")

