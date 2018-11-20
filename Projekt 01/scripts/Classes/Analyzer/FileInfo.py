import scipy.io.wavfile


class FileInfo:
    """ Class holding file data as well as sampling frequency and file name """

    def __init__(self, file_name):
        """
        :param file_name:   File name
        """
        self.file_name = file_name
        self.fs, self.data = scipy.io.wavfile.read(self.file_name)
