#!/usr/bin/env python3
# -*- coding: utf-8 -*-



def sweep_5_measures(W,truth_Y):
    measures = np.zeros((1000,6))
    
    contingency= []
    
    for r in np.arange(1000):   
        
        retain_W = W[:,0:r+1]
        predict_Y = KMeans(retain_W)

        measures[r] = [r+1 , homogeneity(truth_Y, predict_Y),\
        completeness(truth_Y, predict_Y),\
        V_measure(truth_Y, predict_Y),\
        adjusted_rand(truth_Y, predict_Y),\
        adjusted_mutual_info(truth_Y, predict_Y)]
    
        contingency.append(contingency_matrix(truth_Y, predict_Y))

    return measures,contingency

def measures_NMF(n):
    from sklearn.decomposition import NMF
    nmf = NMF(n_components=n, init='random', random_state=0)
    retain_W = nmf.fit_transform(X_tfidf_3)
    predict_Y = KMeans(retain_W)
    measures = [n , homogeneity(truth_Y, predict_Y),\
    completeness(truth_Y, predict_Y),\
    V_measure(truth_Y, predict_Y),\
    adjusted_rand(truth_Y, predict_Y),\
    adjusted_mutual_info(truth_Y, predict_Y)]
    contingency = contingency_matrix(truth_Y, predict_Y)

    return measures,contingency

measures_SVD,contingency_SVD = sweep_5_measures(W_svd_3,truth_Y)

for n in [1,2,3,5,10,20,50,100,300]:
    print(measures_NMF(n))



