

# Norm
W_nmf_3_best_norm = W_nmf_3_best/W_nmf_3_best.std(axis=0)
W_svd_3_best_norm = W_svd_3_best/W_svd_3_best.std(axis=0)


# Logarithm
W_nmf_3_best_log = np.log( W_nmf_3_best + 1e-3 )


# Logarithm + Norm
W_nmf_3_best_log_norm = W_nmf_3_best_log/W_nmf_3_best_log.std(axis=0)


# Norm + Logarithm
W_nmf_3_best_norm_log = np.log(W_nmf_3_best_norm + 1e-3)


