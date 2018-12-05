import glob
import numpy as np
import sklearn.model_selection
import datetime

from Classes.Analyzer import *


def flatten(iterable):
    return [y for x in iterable for y in x]


def get_train_files():
    """ Get list of wav files in the train directory """
    return glob.glob('../train/*.wav')


def get_eval_files():
    return glob.glob('../eval/*.wav')


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


def get_gmm_models(data, cross_validation=True):
    """ Get GMM models for each of numbers """

    if cross_validation:
        kf = sklearn.model_selection.KFold(n_splits=len(data) // 2)

        all_models = []
        all_tests_data = []

        for train_idx, test_idx in kf.split(data):
            train_data = []
            for idx in train_idx:
                train_data.append(data[list(data.keys())[idx]])

            test_data = []
            for idx in test_idx:
                test_data.append(data[list(data.keys())[idx]])

            train_data = flatten(train_data)
            test_data = flatten(test_data)

            all_tests_data.append(test_data)

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

        return all_models, all_tests_data
    else:
        models = []

        train_data = flatten(data.values())

        for i in range(10):
            combined_mfcc = None
            current_num_params = list(filter(lambda p: p.get_rec_num() == i, train_data))

            for param in current_num_params:
                if combined_mfcc is None:
                    combined_mfcc = param.mfcc
                else:
                    combined_mfcc = np.concatenate((combined_mfcc, param.mfcc))

            models.append(GmmModel(combined_mfcc, i))

        return models


def get_recognition_ratio(all_models, all_tests, suffix=''):
    results_file = open('../results/results_{}{}.txt'.format(datetime.datetime.now().strftime("%H_%M_%S"), suffix), 'w+')
    results_file.truncate(0)

    all_count = sum([len(x) for x in all_tests])
    correct_count = 0

    for models, tests in zip(all_models, all_tests):
        for entry in tests:
            logs = []
            nums = []

            for model in models:
                logs.append(model.gmm.score(entry.mfcc))
                nums.append(model.number)

            idx = logs.index(max(logs))

            if nums[idx] == entry.get_rec_num():
                correct_count += 1

            results_file.write('{},{},{:.2f}\n'.format(entry.file_info.file_name[9:], nums[idx], max(logs)))

    rr = correct_count / all_count

    results_file.write('RR: {:.2f}'.format(rr))
    print(rr)
    results_file.close()

    return rr


def recognize(models, tests):
    results_file = open('../results/results_with_tests_{}.txt'.format(datetime.datetime.now().strftime("%H_%M_%S")), 'w+')
    results_file.truncate(0)

    for test in tests:
        logs = []
        nums = []

        for model in models:
            logs.append(model.gmm.score(test.mfcc))
            nums.append(model.number)

        idx = logs.index(max(logs))
        results_file.write('{},{},{:.2f}\n'.format(test.file_info.file_name[8:], nums[idx], max(logs)))

    results_file.close()


def main():
    # files = get_train_files()
    # eval_files = get_eval_files()
    # speaker_data = get_speaker_data(files)
    # # all_models, all_tests_data = get_gmm_models(speaker_data)  # Models for all of the numbers from (0-9)
    # # get_recognition_ratio(all_models, all_tests_data, '_cov_spherical')
    #
    # models = get_gmm_models(speaker_data, cross_validation=False)
    # tests = get_speaker_data(eval_files)
    # print('dupa')
    # recognize(models, flatten(tests.values()))

    # Testing

    from eval import evaluate


    files = get_train_files()
    eval_files = get_eval_files()
    speaker_data = get_speaker_data(files)
    all_models, all_tests_data = get_gmm_models(speaker_data)
    models = get_gmm_models(speaker_data, cross_validation=False)
    tests = get_speaker_data(eval_files)
    #recognize(models, flatten(tests.values()))
    evaluate('dupa.txt')



if __name__ == "__main__":
    main()
