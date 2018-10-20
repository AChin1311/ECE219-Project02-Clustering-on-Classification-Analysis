#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def Q4b_plot(processed_W,predict_Y_best_i):
        
    plt.figure()
    for i in range(processed_W.shape[0]):
        if predict_Y_best_i[i] == 1:
            plt.scatter(processed_W[i,0],processed_W[i,1],c = 'b' )
        else:
            plt.scatter(processed_W[i,0],processed_W[i,1],c = 'g' )
    plt.show()




# 4(a)
processed_W = W_svd_3_best
predict_Y_best_i = KMeans(processed_W)
Q4b_plot(processed_W,predict_Y_best_i)
Q4b_plot(processed_W,truth_Y)


processed_W = W_nmf_3_best
predict_Y_best_i = KMeans(processed_W)
Q4b_plot(processed_W,predict_Y_best_i)
Q4b_plot(processed_W,truth_Y)



# 4(b)

processed_W = W_svd_3_best_norm
predict_Y_best_i = KMeans(processed_W)
Q4b_plot(processed_W,predict_Y_best_i)
Q4b_plot(processed_W,truth_Y)

processed_W = W_nmf_3_best_norm
predict_Y_best_i = KMeans(processed_W)
Q4b_plot(processed_W,predict_Y_best_i)
Q4b_plot(processed_W,truth_Y)

processed_W = W_nmf_3_best_log
predict_Y_best_i = KMeans(processed_W)
Q4b_plot(processed_W,predict_Y_best_i)
Q4b_plot(processed_W,truth_Y)

processed_W = W_nmf_3_best_log_norm
predict_Y_best_i = KMeans(processed_W)
Q4b_plot(processed_W,predict_Y_best_i)
Q4b_plot(processed_W,truth_Y)

processed_W = W_nmf_3_best_norm_log
predict_Y_best_i = KMeans(processed_W)
Q4b_plot(processed_W,predict_Y_best_i)
Q4b_plot(processed_W,truth_Y)






