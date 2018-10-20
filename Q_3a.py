#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


def var_ratio(X_tfidf,W): 
    var_tot = X_tfidf.toarray().T.dot(X_tfidf.toarray()).trace()
    ratio = np.zeros((1000,2))

    for r in np.arange(1000):   
        X_retain = W[:,0:r+1]
        var_retain = np.trace( X_retain.T.dot(X_retain))
        var_ratio = var_retain / var_tot   
        ratio[r] = [r+1,var_ratio]
        print(r+1)
    return ratio


var_ratio_svd = var_ratio(X_tfidf_3,W_svd_3)
#var_ratio_nmf = var_ratio(X_tfidf_3,W_nmf_3)
        
        



# Plot the r vs retain percentage

plt.figure()
plt.plot(var_ratio_svd[:,0],var_ratio_svd[:,1])
plt.title("LSI")
plt.xlabel("r")
plt.ylabel("Retain ratio")
plt.imshow()


#plt.figure()
#plt.plot(var_ratio_nmf[:,0],var_ratio_nmf[:,1])
#plt.title("NMF")
#plt.xlabel("r")
#plt.ylabel("Retain percentage")
#plt.imshow()
#
#



