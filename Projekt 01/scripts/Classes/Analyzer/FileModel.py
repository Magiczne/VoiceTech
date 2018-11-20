import sklearn.mixture

from .FileParameters import FileParameters
from Classes import Config


class FileModel:
    """ Class holding GMM model for the file """

    def __init__(self, file_parameters, n_components, n_iter):
        """
        :type file_parameters:      FileParameters
        :param file_parameters:     File parameters object
        """
        self.parameters = file_parameters
        self.n_components = n_components
        self.n_iter = n_iter

        config = Config()

        mixture = sklearn.mixture.GaussianMixture(
            n_components=config.get_param('GMM', 'Components', int),
            max_iter=config.get_param('GMM', 'Iterations', int),
            covariance_type='diag'
        )

        self.gmm = mixture.fit(file_parameters.mfcc)
