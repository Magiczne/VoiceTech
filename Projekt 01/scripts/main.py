import glob
import numpy as np
from Classes.Analyzer import *


def get_wav_files():
    """ Get list of wav files in the train directory """
    return glob.glob('../train/*.wav')


def get_gmm_models(files):
    """ Get GMM models for each of numbers """
    params = []
    for file in files:
        info = FileInfo(file)
        params.append(FileParameters(info))

    models = []

    for i in range(10):
        combined_mfcc = None
        current_num_params = list(filter(lambda p: p.get_rec_num() == i, params))

        for param in current_num_params:
            if combined_mfcc is None:
                combined_mfcc = param.mfcc
            else:
                combined_mfcc = np.concatenate((combined_mfcc, param.mfcc))

        models.append(GmmModel(combined_mfcc, i))

    return models


def main():
    files = get_wav_files()
    models = get_gmm_models(files)


if __name__ == "__main__":
    main()
