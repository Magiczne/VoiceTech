from python_speech_features import mfcc
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plot
import numpy


class Lab3:
    def __init__(self, file_a, file_i):
        self.file_a = file_a
        self.file_i = file_i

    # region MFCC

    @staticmethod
    def calculate_mfcc(file):
        fs, signal = wavfile.read(file)
        return mfcc(signal, fs, winfunc=numpy.hamming)

    @staticmethod
    def plot_c1_c2(mfcc_a, mfcc_i):
        fig, ((ax_a1, ax_a2), (ax_i1, ax_i2)) = plot.subplots(2, 2, tight_layout=True)

        # C1 and C2
        a_c1 = [item[0] for item in mfcc_a]
        a_c1_x = mfcc_a[:, 0]


        a_c2 = [item[1] for item in mfcc_a]

        i_c1 = [item[0] for item in mfcc_i]
        i_c2 = [item[1] for item in mfcc_i]

        # Plots
        ax_a1.hist(a_c1)
        ax_a2.hist(a_c2)

        ax_i1.hist(i_c1)
        ax_i2.hist(i_c2)

        ax_a1.set_title('C1 - aaa')
        ax_a2.set_title('C2 - aaa')
        ax_i1.set_title('C1 - iii')
        ax_i2.set_title('C2 - iii')

        plot.show()

    def mfcc_stats(self, mfcc_a, mfcc_i):
        # C1 and C2
        a_c1 = [item[0] for item in mfcc_a]
        a_c2 = [item[1] for item in mfcc_a]

        i_c1 = [item[0] for item in mfcc_i]
        i_c2 = [item[1] for item in mfcc_i]

        a_c1_avg = numpy.mean(a_c1)
        a_c2_avg = numpy.mean(a_c2)

        i_c1_avg = numpy.mean(i_c1)
        i_c2_avg = numpy.mean(i_c2)

        print("Aaa: ")
        print("C1 mean: {}\nC2 mean: {}".format(a_c1_avg, a_c2_avg))

        print("Iii: ")
        print("C1 mean: {}\nC2 mean: {}".format(a_c2_avg, i_c2_avg))

    def do_mfcc(self):
        mfcc_a = self.calculate_mfcc(self.file_a)
        mfcc_i = self.calculate_mfcc(self.file_i)

        self.plot_c1_c2(mfcc_a, mfcc_i)
        self.mfcc_stats(mfcc_a, mfcc_i)

    # endregion


def main():
    lab = Lab3('../audio/aaa_16khz.wav', '../audio/iii_16khz.wav')

    lab.do_mfcc()


if __name__ == "__main__":
    main()
