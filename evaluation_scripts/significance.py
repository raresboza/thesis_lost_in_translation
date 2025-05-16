import itertools

from scipy.stats import wilcoxon, ttest_rel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.multitest import multipletests

def wilcoxon_test(scores, alpha=0.01):
    n = len(scores)
    p_matrix = np.ones((n, n))

    correction = (n * (n - 1)) / 2

    for i, j in itertools.combinations(range(n), 2):
        stat, p = wilcoxon(scores[i], scores[j])
        print(p)
        p_corrected = min(p * correction, 1.0) # Bonferroni correction

        p_matrix[i, j] = p_corrected
        p_matrix[j, i] = p_corrected

    print(p_matrix)
    sig_matrix = (p_matrix < alpha).astype(float)
    np.fill_diagonal(sig_matrix, np.nan)
    print(sig_matrix)

    return sig_matrix