from python_speech_features import mfcc
import scipy.io.wavfile as wavfile
import scipy.signal
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plot
import numpy as np


def butter_lowpass(cutoff, fs, order=5):
    """ https://gist.github.com/junzis/e06eca03747fc194e322 """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    """ https://gist.github.com/junzis/e06eca03747fc194e322 """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


class Lab3:
    def __init__(self, file_a, file_i):
        self.file_a = file_a
        self.file_i = file_i

    # region MFCC

    @staticmethod
    def calculate_mfcc(file):
        fs, signal = wavfile.read(file)
        return mfcc(signal, fs, winfunc=np.hamming)

    @staticmethod
    def plot_c1_c2(mfcc_a, mfcc_i):
        fig, (ax1, ax2) = plot.subplots(2, 2, tight_layout=True)

        for i in range(2):
            # a_coefficients = mfcc_a[i, :]
            # i_coefficients = mfcc_i[i, :]

            a_coefficients = mfcc_a[:, i]
            i_coefficients = mfcc_i[:, i]

            ax1[i].hist(a_coefficients)
            ax2[i].hist(i_coefficients)

            ax1[i].set_title("C{} - aaa".format(i + 1))
            ax2[i].set_title("C{} - iii".format(i + 1))

        plot.show()

    @staticmethod
    def mfcc_stats(mfcc_a, mfcc_i):
        a_coefficients_avg = []
        i_coefficients_avg = []

        for i in range(2):
            a_coefficients = mfcc_a[:, i]
            i_coefficients = mfcc_i[:, i]

            a_coefficients_avg.append(np.mean(a_coefficients))
            i_coefficients_avg.append(np.mean(i_coefficients))

        print("Aaa: ")
        print("C1 mean: {}\nC2 mean: {}".format(a_coefficients_avg[0], a_coefficients_avg[1]))

        print("Iii: ")
        print("C1 mean: {}\nC2 mean: {}".format(i_coefficients_avg[0], i_coefficients_avg[1]))

    def laryngeal_tone(self):
        win_length = 25
        fs = 16000
        window = scipy.signal.get_window('hamming', int(fs * (win_length / 1000)))

        fig, axes = plot.subplots(1, 2, tight_layout=True)

        signal_data = [
            wavfile.read(self.file_a),
            wavfile.read(self.file_i)
        ]
        titles = [
            '25ms - aaa',
            '25ms - iii'
        ]

        for i in range(2):
            data = np.frombuffer(signal_data[i][1], np.int16)
            f, t, Zxx = scipy.signal.stft(data, fs, window, nperseg=len(window))

            axes[i].pcolormesh(t, f, np.abs(Zxx))
            axes[i].set_title(titles[i])
            axes[i].set_xlabel('Czas [s]')
            axes[i].set_ylabel('Częstotliwość [Hz]')

            data = butter_lowpass_filter(data, 200, 16000)

            # Autocorrelation
            r = np.correlate(data, data, mode='full')[len(data) - 1:]

            peaks, _ = scipy.signal.find_peaks(r)

            distance = np.mean(np.diff(peaks))
            T = distance / fs
            f = 1 / T

            plot.figure()
            plot.plot(r)
            # plot.plot(peaks, r[peaks], "x")
            plot.xlabel('Lag (samples)')
            plot.title("f0 = {}".format(f))
            # plot.xlim(0, 1500)

        plot.show()

    def do_mfcc(self):
        mfcc_a = self.calculate_mfcc(self.file_a)
        mfcc_i = self.calculate_mfcc(self.file_i)

        self.plot_c1_c2(mfcc_a, mfcc_i)
        self.mfcc_stats(mfcc_a, mfcc_i)
        self.laryngeal_tone()

    # endregion


def main():
    lab = Lab3('../audio/aaa_16khz.wav', '../audio/iii_16khz.wav')

    lab.do_mfcc()


if __name__ == "__main__":
    main()
