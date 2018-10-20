#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs



# Using best r

r_best_svd = 2
r_best_nmf = 2



W_svd_3_best = W_svd_3[:,0:r_best_svd]
W_nmf_3_best = W_nmf_3[:,0:r_best_nmf]



predict_Y_best_svd = KMeans(W_svd_3_best)
predict_Y_best_nmf = KMeans(W_nmf_3_best)



# Plot SVD scatter

#plt.figure(1)
#
#for i in range(len(predict_Y_best_svd)):
#    if predict_Y_best_svd[i] == 1:
#        plt.scatter(W_svd_3_best[i,0],W_svd_3_best[i,1],c = 'b' )
#    else:
#        plt.scatter(W_svd_3_best[i,0],W_svd_3_best[i,1],c = 'g' )
#plt.show()
#
#
#
## Plot NMF scatter
#
#
#
#plt.figure(1)
#
#for i in range(len(predict_Y_best_nmf)):
#    if predict_Y_best_nmf[i] == 1:
#        plt.scatter(W_nmf_3_best[i,0],W_nmf_3_best[i,1],c = 'b' )
#    else:
#        plt.scatter(W_nmf_3_best[i,0],W_nmf_3_best[i,1],c = 'g' )
#plt.show()




