import glob
from Classes.Analyzer import *


def get_wav_files():
    """ Get list of wav files in the train directory """
    return glob.glob('../train/*.wav')


def get_gmm_models(files):
    models = []
    for file in files:
        info = FileInfo(file)
        params = FileParameters(info)
        model = FileModel(params, 8, 100)
        models.append(model)

    return models


def main():
    files = get_wav_files()
    models = get_gmm_models(files)


if __name__ == "__main__":
    main()
