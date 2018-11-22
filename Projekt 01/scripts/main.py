import glob
import numpy as np
import sklearn.model_selection

from Classes.Analyzer import *


def get_wav_files():
    """ Get list of wav files in the train directory """
    return glob.glob('../train/*.wav')


def get_speaker_data(files):
    """ Get speaker data """
    speaker_data = {}

    for file in files:
        info = FileInfo(file)
        param = FileParameters(info)

        if param.get_speaker() not in speaker_data:
            speaker_data[param.get_speaker()] = []

        speaker_data[param.get_speaker()].append(param)

    return speaker_data


def get_gmm_models(data):
    """ Get GMM models for each of numbers """

    kf = sklearn.model_selection.KFold(n_splits=len(data) // 2)

    all_models = []

    for train_idx, test_idx in kf.split(data):
        # print("TRAIN:", train_idx, "TEST:", test_idx)

        train_data = []
        for idx in train_idx:
            train_data.append(data[list(data.keys())[idx]])

        train_data = [y for x in train_data for y in x]

        models = []
        for i in range(10):
            combined_mfcc = None
            current_num_params = list(filter(lambda p: p.get_rec_num() == i, train_data))

            for param in current_num_params:
                if combined_mfcc is None:
                    combined_mfcc = param.mfcc
                else:
                    combined_mfcc = np.concatenate((combined_mfcc, param.mfcc))

            models.append(GmmModel(combined_mfcc, i))

        all_models.append(models)

    return all_models


def main():
    files = get_wav_files()
    speaker_data = get_speaker_data(files)
    all_models = get_gmm_models(speaker_data)  # Models for all of the numbers from (0-9)

    print('dupa')



if __name__ == "__main__":
    main()
