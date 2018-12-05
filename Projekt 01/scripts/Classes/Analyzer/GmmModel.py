import sklearn.mixture

from .FileParameters import FileParameters
from .. import Config


class GmmModel:
    """ Class holding GMM model for the file """

    def __init__(self, mfcc, number):
        """
        :type file_parameters:      FileParameters
        :param file_parameters:     File parameters object
        """
        self.mfcc = mfcc
        self.number = number

        config = Config()

        mixture = sklearn.mixture.GaussianMixture(
            n_components=config.get_param('GMM', 'Components', int),
            max_iter=config.get_param('GMM', 'Iterations', int),
            covariance_type=config.get_param('GMM', 'Covariance', str),
            random_state=3
        )

        self.gmm = mixture.fit(self.mfcc)
