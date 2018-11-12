import python_speech_features
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import sklearn.mixture as mixture
from scipy.stats import norm


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

    @staticmethod
    def gaussian(x, mu, sigma):
        return norm.pdf(x, loc=mu, scale=sigma)

    def plot_mixture_model(self, mfcc, gmm, m):
        """
        :type gmm:      mixture.GaussianMixture
        :param mfcc:    MFCC data
        :param gmm:     GMM data
        :param m:       MFCC Coefficient number (from 0)
        :return:
        """
        c_m = mfcc[:, m]

        plt.hist(c_m)
        plt.title('Dopasowanie dla współczynnika C{}'.format(m))

        space = np.linspace(min(c_m), max(c_m))
        gauss_sum = np.zeros(len(space))

        for i in range(self.n_components):
            mean = gmm.means_[i, m]
            var = np.sqrt(gmm.covariances_[i, m])
            weight = gmm.weights_[i]

            gauss = weight * self.gaussian(space, mean, var)
            gauss_sum += gauss
            plt.plot(space, gauss * 450, label='Krzywa {}'.format(i + 1))

        plt.plot(space, gauss_sum * 450, label='Sumaryczne dopasowanie')
        plt.legend()

        plt.show()

        print("Score: {}".format(gmm.score(mfcc)))

        return gmm.aic(mfcc)
