import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plot
import scipy.io.wavfile
import scipy.signal


filename = '../audio/MojaNajdrozsza_16bit_PCM.wav'


class AudioFile:
    """ https://stackoverflow.com/a/6951154/7101876 """
    chunk = 1024

    def __init__(self, file):
        """ Init audio stream """
        self.file = file
        self.waveform = wave.open(self.file, 'rb')
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.get_format(),
            channels=self.waveform.getnchannels(),
            rate=self.waveform.getframerate(),
            output=True
        )

    # region Getters

    def get_data(self):
        """ Get samples and sample rate """
        return scipy.io.wavfile.read(self.file)

    def get_format(self):
        return self.audio.get_format_from_width(self.waveform.getsampwidth())

    # endregion

    # region Playing file

    def play(self):
        """ Play entire file in chunks"""
        data = self.waveform.readframes(self.chunk)
        while len(data) > 0:
            self.stream.write(data)
            data = self.waveform.readframes(self.chunk)

    # endregion

    # region Plots

    def oscillogram(self):
        fs, signal = self.get_data()
        signal = np.fromstring(signal, np.int16)

        time_axis = np.linspace(0, len(signal) / fs, num=len(signal))

        plot.figure(1)
        plot.title(filename)
        plot.plot(time_axis, signal)
        plot.xlabel('Czas [s]')
        plot.ylabel('Amplituda')
        plot.show()

    def stft(self):
        fs, signal = self.get_data()
        f, t, Zxx = scipy.signal.stft(signal, fs)

        plot.figure(2)
        plot.pcolormesh(t, f, np.abs(Zxx))
        plot.xlabel('Czas [s]')
        plot.ylabel('Częstotliwość [Hz]')
        plot.show()

    def windows(self):
        fs, signal = self.get_data()

        fig, axes = plot.subplots(2, 3, tight_layout=True)

        # Window lengths in [ms]
        win_lengths = [5, 10, 20, 50, 100, 200]

        for i in range(6):
            window = scipy.signal.get_window('hamming', int(fs * (win_lengths[i] / 1000)))
            f, t, Zxx = scipy.signal.stft(signal, fs, window, nperseg=len(window))

            ax = axes[i // 3 - 1, i % 3]
            ax.pcolormesh(t, f, np.abs(Zxx))
            ax.set_title("Okno Hamminga {} ms".format(win_lengths[i]))
            ax.set_xlabel('Czas [s]')
            ax.set_ylabel('Częstotliwość [Hz]')

        plot.show()

    # endregion

    # region Cleaning up

    def close(self):
        """ Graceful shutdown """
        self.stream.close()
        self.audio.terminate()
        self.waveform.close()

    # endregion


def main():
    file = AudioFile(filename)
    # file.play()

    # file.oscillogram()
    # file.stft()
    file.windows()

    file.close()


if __name__ == "__main__":
    main()
