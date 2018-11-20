import scipy.io.wavfile
import os
import python_speech_features
import numpy as np
import sklearn.mixture as mixture

files_names = []
for root, dirs, files in os.walk(os.path.abspath('.train/..')):
    for file in files:
        if file.endswith('.wav'):
            files_names.append(file)


class FileInfo:

    def __init__(self, file_name):
        self.file_name = file_name
        self.fs, self.data = scipy.io.wavfile.read(self.file_name)


class Parameters:
    """

    :type file_info: FileInfo
    """

    def __init__(self, file_info):
        self.file_info = file_info
        self.mfcc = python_speech_features.mfcc(file_info.data, file_info.fs, winfunc=np.hamming)

    def get_speaker(self):
        return self.file_info.file_name[:-7]

    def get_rec_num(self):
        return self.file_info.file_name[6]


class GMM:
    """

    :type parameters: Parameters
    """
    def __init__(self, parameters, n_components, n_iter):
        self.parameters = parameters
        self.n_components = n_components
        self.n_iter = n_iter
        self.gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='diag', max_iter=n_iter).fit(parameters.mfcc)


for file in files_names:
    f = FileInfo(file)
    param = Parameters(f)
    gmm = GMM(param, 8, 10)


