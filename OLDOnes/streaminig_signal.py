import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import subprocess
import os
from matplotlib.animation import FuncAnimation
# from scipy.fftpack import fft
import scipy.signal
# plt.style.use('fivethirtyeight')
wholerun = [0] * 40000
rms1 = 0

def animate(i):
    proc_args = ['arecord', '-D', 'plughw:1,0', '-d', '1', '-c1', '-M', '-r', '48000', '-f', 'S32_LE', '-t', 'wav',
                 '-V', 'mono', '-v', 'input_read1.wav']
    rec_proc = subprocess.Popen(proc_args, shell=False, preexec_fn=os.setsid)
    print("startRecordingArecord()> rec_proc pid= " + str(rec_proc.pid))

    # read the input file
    Fs, audio = wavfile.read('../input_read1.wav', mmap=True)

    # antyaliasing filter
    filtr = scipy.signal.firwin2(1024, [0, 0.167, 0.183, 1], [1, 1, 0, 0])
    audio = scipy.signal.convolve(audio, filtr, mode='full', method='auto')
    audio = audio[:Fs]

    # flitr przeciwkoprzydzwiekowi
    filtr = scipy.signal.firwin(1023, 160, fs=Fs, pass_zero=False)
    audio = scipy.signal.convolve(audio, filtr, mode='full', method='auto')
    audio = audio[:Fs]

    # take every 6 sample to reduce sampling rate
    audio1 = audio[0::6]
    Fs /= 6

    # normalization
    rms = np.sqrt(np.mean(audio1 ** 2))
    global rms1
    rms1 = 0.8 * rms1 + 0.2 * rms
    audio1 = audio1 / rms1
    audio2 = np.append(audio1[0], audio1[1:] - 0.95 * audio1[:-1])

    global wholerun
    wholerun = wholerun[8000:]
    wholerun = np.append(wholerun, audio2)

    # time axis
    x_values = np.arange(0, len(wholerun), 1) / float(Fs)
    # to secs
    x_values *= 1000

    plt.cla()
    plt.plot(x_values, wholerun, label='Signal')
    plt.legend(loc='upper left')
    plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.tight_layout()
plt.show()