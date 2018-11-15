from GmmAnalyzer import GmmAnalyzer
import numpy as np
import sklearn.mixture
from functools import reduce
import matplotlib.pyplot as plt

"""
score -> log(P(MFCC_A|GMM_A))
p(GMM_A|MFCC_A) = ( p(MFCC_A|GMM_A) * p(GMM_A) ) / p(MFCC_A) 
p(MFCC_A) = p(MFCC_A|GMM_A) * p(GMM_A) + p(MFCC_A|GMM_I) * p(GMM_I)
p(GMM_A) -> prawdopodobieństwo wystąpienia cechy
"""

analyzer_a = GmmAnalyzer('../audio/aaa_16khz.wav')
analyzer_i = GmmAnalyzer('../audio/iii_16khz.wav')
mfcc_a = analyzer_a.get_mfcc()
mfcc_i = analyzer_i.get_mfcc()
gmm_a: sklearn.mixture.GaussianMixture = analyzer_a.get_gmm(mfcc_a, 8, 100)
gmm_i: sklearn.mixture.GaussianMixture = analyzer_i.get_gmm(mfcc_i, 8, 100)

# print(analyzer_a.plot_mixture_model(mfcc_a, analyzer_a.get_gmm(mfcc_a, 1, 1), 1))
# print(analyzer_a.plot_mixture_model(mfcc_a, analyzer_a.get_gmm(mfcc_a, 3, 3), 1))
# print(analyzer_a.plot_mixture_model(mfcc_a, analyzer_a.get_gmm(mfcc_a, 5, 5), 1))
# print(analyzer_a.plot_mixture_model(mfcc_a, analyzer_a.get_gmm(mfcc_a, 7, 15), 1))

# 1.3
# 1.4.1 - Przeanalizuj wartośc kryterium od liczby iteracji
# 1.3       -
# 1.4.1     - 7 iteracji wygląda już dość spoko
# aic = []
# bic = []
#
# plt.figure(figsize=(15, 21))

# for i in range(10, 90, 10):
#     plt.subplot(4, 2, i // 10)
#     data = analyzer_a.plot_mixture_model(mfcc_a, analyzer_a.get_gmm(mfcc_a, 8, i), 1, plot=True, plot_show=False,
#                                           legend=False)

#
# plt.savefig('../charts/1.4.1_2.png')

# for i in [1, 3, 5, 7, 9, 10, 20, 30, 50, 70, 90]:
#     data = analyzer_a.plot_mixture_model(mfcc_a, analyzer_a.get_gmm(mfcc_a, 8, i), 1, plot=True, plot_show=False,
#                                          legend=False)
#     aic.append(data['aic'])
#     bic.append(data['bic'])
#
# for a in aic:
#     print("{}".format(a))
# print('\n')
# for b in bic:
#     print("{}".format(b))


# min_aic = min(aic)
# min_bic = min(bic)
# aic.remove(min_aic)
# bic.remove(min_bic)
#
# aic_likehood = []
# bic_likehood = []
#
# for item in aic:
#     aic_likehood.append(np.exp((min_aic - item) / 2))
#
# for item in bic:
#     bic_likehood.append(np.exp((min_bic - item) / 2))
#
# print(max(aic_likehood), aic_likehood.index(max(aic_likehood)))
# print(max(bic_likehood), bic_likehood.index(max(bic_likehood)))

# 1.4.2 - Ile komponentów jest optymalnych dla modelowania głosek
# 6 jest juz ok
# plt.figure(figsize=(15, 21))
#
# for i in range(1, 9):
#     plt.subplot(4, 2, i)
#     analyzer_a.plot_mixture_model(mfcc_a, analyzer_a.get_gmm(mfcc_a, i, 100), 1, plot_show=False)
#
# plt.savefig('../charts/1.4.2.png')

# 2.1
log_p_mfcc_a_gmm_a = gmm_a.score(mfcc_a)      # log(p(MFCC_A | GMM_A))
log_p_mfcc_i_gmm_a = gmm_a.score(mfcc_i)      # log(p(MFCC_I | GMM_A))
log_p_mfcc_i_gmm_i = gmm_i.score(mfcc_i)      # log(p(MFCC_I | GMM_I))
log_p_mfcc_a_gmm_i = gmm_i.score(mfcc_a)      # log(p(MFCC_A | GMM_I))

print("log(p(MFCC_A | GMM_A)) = {}, {}".format(log_p_mfcc_a_gmm_a, np.exp(log_p_mfcc_a_gmm_a)))
print("log(p(MFCC_I | GMM_A)) = {}, {}".format(log_p_mfcc_i_gmm_a, np.exp(log_p_mfcc_i_gmm_a)))
print("log(p(MFCC_I | GMM_I)) = {}, {}".format(log_p_mfcc_i_gmm_i, np.exp(log_p_mfcc_i_gmm_i)))
print("log(p(MFCC_A | GMM_I)) = {}, {}".format(log_p_mfcc_a_gmm_i, np.exp(log_p_mfcc_a_gmm_i)))


# 2.1.1
def calc_p_mfcc_y(scores):
    ret = 0
    for score in scores:
        ret += np.exp(score) * 0.5
    return ret


def p_gmm_x_mfcc_y(log_p_mfcc_x_gmm_y, scores, p_gmm_x=0.5):
    p_mfcc_x_gmm_y = np.exp(log_p_mfcc_x_gmm_y)
    # print("p_mfcc_x_gmm_y = {}".format(p_mfcc_x_gmm_y))

    return (p_mfcc_x_gmm_y * p_gmm_x) / calc_p_mfcc_y(scores)


p_gmm_a_mfcc_a = p_gmm_x_mfcc_y(log_p_mfcc_a_gmm_a, [gmm_a.score(mfcc_a), gmm_i.score(mfcc_a)])     # p(GMM_A/MFCC_A)
p_gmm_a_mfcc_i = p_gmm_x_mfcc_y(log_p_mfcc_i_gmm_a, [gmm_a.score(mfcc_a), gmm_i.score(mfcc_a)])     # p(GMM_A/MFCC_I)
p_gmm_i_mfcc_a = p_gmm_x_mfcc_y(log_p_mfcc_a_gmm_i, [gmm_a.score(mfcc_i), gmm_i.score(mfcc_i)])     # p(GMM_I/MFCC_A)
p_gmm_i_mfcc_i = p_gmm_x_mfcc_y(log_p_mfcc_i_gmm_i, [gmm_a.score(mfcc_i), gmm_i.score(mfcc_i)])     # p(GMM_I/MFCC_I)

print("p(GMM_A/MFCC_A) = {}".format(p_gmm_a_mfcc_a))
print("p(GMM_A/MFCC_I) = {}".format(p_gmm_a_mfcc_i))
print("p(GMM_I/MFCC_A) = {}".format(p_gmm_i_mfcc_a))
print("p(GMM_I/MFCC_I) = {}".format(p_gmm_i_mfcc_i))
