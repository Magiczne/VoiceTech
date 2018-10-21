import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plot
import scipy.io.wavfile


filename = '../MojaNajdrozsza_16bit_PCM.wav'


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

    def show_oscillogram(self):
        fs, signal = self.get_data()
        signal = np.fromstring(signal, np.int16)

        time_axis = np.linspace(0, len(signal) / fs, num=len(signal))

        plot.figure(1)
        plot.title(filename)
        plot.plot(time_axis, signal)
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
    file.play()

    file.show_oscillogram()

    file.close()


if __name__ == "__main__":
    main()
