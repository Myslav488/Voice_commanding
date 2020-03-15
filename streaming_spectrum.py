import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import subprocess
import os
from matplotlib.animation import FuncAnimation
from scipy.fftpack import fft
import scipy.signal


def animate(i):
    proc_args = ['arecord', '-D', 'plughw:1,0', '-d', '1', '-c1', '-M', '-r', '48000', '-f', 'S32_LE', '-t', 'wav',
                 '-V', 'mono', '-v', 'input_read1.wav']
    rec_proc = subprocess.Popen(proc_args, shell=False, preexec_fn=os.setsid)
    print("startRecordingArecord()> rec_proc pid= " + str(rec_proc.pid))

    # read the input file
    Fs, audio = wavfile.read('input_read1.wav', mmap=True)

    # antyaliasing filter
    filtr = scipy.signal.firwin2(1024, [0, 0.167, 0.183, 1], [1, 1, 0, 0])
    audio = scipy.signal.convolve(audio, filtr, mode='full', method='auto')
    audio = audio[:Fs]

    # flitr przeciwkoprzydzwiekowi
    filtr = scipy.signal.firwin(1023, 160, fs=Fs, pass_zero=False)
    audio = scipy.signal.convolve(audio, filtr, mode='full', method='auto')
    audio = audio[:Fs]

    # print(len(audio))
    # take evry 6 sample to reduce sampling rate
    audio1 = audio[0::6]
    Fs /= 6

    # normalization
    rms = np.sqrt(np.mean(audio1 ** 2))
    audio1 = audio1 / rms
    audio2 = np.append(audio1[0], audio1[1:] - 0.95 * audio1[:-1])

    audiofft = fft(audio2)
    audiofft = (2 / Fs) * np.abs(audiofft[:int(Fs) // 2])

    ##Filtracja dzwieku wentylatora
    # zapisywanie widma sygnalu
    # if os.path.exists("widmo.csv"):
    #    os.remove("widmo.csv")
    # audiofft.tofile("widmo.csv", sep=';', format='%10.5f')

    # wczytywanie widma sygnalu
    # audiox = np.fromfile("widmo.csv", sep=';')
    # print(np.info(audiox))
    # audiofft -= audiox
    # freq axis
    x_values = np.arange(0, len(audiofft), 1)
    plt.cla()
    plt.plot(x_values, audiofft, label='Signal')
    plt.legend(loc='upper left')
    plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.tight_layout()
plt.show()