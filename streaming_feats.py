import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import subprocess
import os
from matplotlib.animation import FuncAnimation
from mfcc_lib import mfcc, logfbank
import scipy.signal

wholerun = np.zeros((13, 495))


def animate(i):
    proc_args = ['arecord', '-D', 'plughw:1,0', '-d', '1', '-c1', '-M', '-r', '48000', '-f', 'S32_LE', '-t', 'wav',
                 '-V', 'mono', '-v', 'input_read1.wav']
    rec_proc = subprocess.Popen(proc_args, shell=False, preexec_fn=os.setsid)
    # print("startRecordingArecord()> rec_proc pid= " + str(rec_proc.pid))

    # read the input file
    Fs, audio = wavfile.read('input_read1.wav', mmap=True)

    # antyaliasing filter
    filtr = scipy.signal.firwin2(1024, [0, 0.167, 0.183, 1], [1, 1, 0, 0])
    audio = scipy.signal.convolve(audio, filtr, mode='full', method='auto')
    audio = audio[:Fs]

    # take evry 6 sample to reduce sampling rate
    audio1 = audio[0::6]
    Fs /= 6

    # normalization
    rms = np.sqrt(np.mean(audio1 ** 2))
    audio1 = audio1 / rms

    mfcc_feat = mfcc(audio1, Fs)
    # fbank_feat = logfbank(audio1,sampling_freq)

    mfcc_feat = mfcc_feat.T
    # mfcc_feat = np.flipud(mfcc_feat)
    global wholerun
    # wholerun = wholerun[,99:]
    wholerun = np.delete(wholerun, range(0, 99), axis=1)
    wholerun = np.append(wholerun, mfcc_feat, axis=1)

    plt.cla()
    plt.imshow(wholerun, extent=(0, 495, 0, 130))
    # print(mfcc_feat.shape)
    # plt.cla()
    # fbank_feat = fbank_feat.T
    # plt.imshow(fbank_feat)
    # plt.tight_layout(pad=1.08, h_pad=None)


ani = FuncAnimation(plt.gcf(), animate, interval=1000)

# plt.tight_layout()
plt.show()
