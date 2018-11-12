import python_speech_features
import scipy.io.wavfile
from scipy.stats import norm
import numpy as np
import sklearn.mixture as mixture
import matplotlib.pyplot as plt

"""
score -> log(P(MFCC_A|GMM_A))
p(GMM_A|MFCC_A) = ( p(MFCC_A|GMM_A) * p(GMM_A) ) / p(MFCC_A) 
p(MFCC_A) = p(MFCC_A|GMM_A) * p(GMM_A) + p(MFCC_A|GMM_I) * p(GMM_I)
p(GMM_A) -> prawdopodobieństwo wystąpienia cechy
"""


class GmmAnalyzer:
    def __init__(self, file):
        self.file = file
        self.n_components = 0

    def get_mfcc(self):
        fs, signal = scipy.io.wavfile.read(self.file)
        return python_speech_features.mfcc(signal, fs, winfunc=np.hamming)

    def get_gmm(self, mfcc, n_components, n_iter):
        self.n_components = n_components
        gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='diag')
        gmm.n_iter_ = n_iter
        return gmm.fit(mfcc)

    def gaussian(self, x, mu, sigma):
        return norm.pdf(x, loc=mu, scale=sigma)

    def plot_magic(self, mfcc, gmm, m):
        """
        :type gmm:      mixture.GaussianMixture
        :param mfcc:    MFCC data
        :param gmm:     GMM data
        :param m:       MFCC Coefficient number (from 0)
        :return:
        """
        c_m = mfcc[:, m]

        plt.hist(c_m)

        for i in range(self.n_components):
            mean = gmm.means_[i, m]
            cov = gmm.covariances_[i, m]
            space = np.linspace(min(c_m), max(c_m))
            gauss = self.gaussian(space, mean, cov)
            plt.plot(gauss)

        plt.show()


analyzer = GmmAnalyzer('../audio/aaa_16khz.wav')
mfcc = analyzer.get_mfcc()
gmm = analyzer.get_gmm(mfcc, 8, 1)
analyzer.plot_magic(mfcc, gmm, 1)