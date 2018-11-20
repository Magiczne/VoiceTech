import sklearn.mixture

from .FileParameters import FileParameters


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

        mixture = sklearn.mixture.GaussianMixture(n_components=n_components, covariance_type='diag', max_iter=n_iter)

        self.gmm = mixture.fit(file_parameters.mfcc)
