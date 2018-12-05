import python_speech_features
import numpy as np

from .FileInfo import FileInfo
from Classes import Config


class FileParameters:
    """ Class holding more specific information about file as well as its MFCC parameters """

    def __init__(self, file_info):
        """
        :type file_info:    FileInfo
        :param file_info:   File info object
        """

        config = Config()

        self.file_info = file_info
        self.mfcc = python_speech_features.mfcc(
            file_info.data,
            file_info.fs,
            appendEnergy=True,
            winfunc=np.hamming,
            winlen=config.get_param('MFCC', 'WinLen', float),
            nfilt=7,
            preemph=1,
            ceplifter=0

        )

    def get_speaker(self):
        """
        Get speaker
        :return:    Speaker code
        """
        return self.file_info.file_name[9:-7]

    def get_rec_num(self):
        """
        Get number that is present on the recording
        :return:    Number on the recording
        """
        return int(self.file_info.file_name[-6])
