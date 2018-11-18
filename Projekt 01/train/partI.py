import wave
import scipy.io.wavfile
import os

files_names = []
for root, dirs, files in os.walk(os.path.abspath('.train/..')):
    for file in files:
        if file.endswith('.wav'):
            files_names.append(file)

class FileInfo:
    def getinfo(files_names):
        j = 0
        files_info = {}
        file = []
        data = [0] * (len(files_names))
        freq = [0] * (len(files_names))
        
        for name in files_names:
            file.append(wave.open(name))
            freq[j], data[j] = scipy.io.wavfile.read(name)
            files_info['File' + str(j + 1)] = name, data[j], freq[j]
            file[j].close()
            j = j + 1
        return files_info

x = FileInfo
fileinfo = x.getinfo(files_names)
