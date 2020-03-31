from matplotlib.animation import FuncAnimation
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import lib_filter as filt
import numpy as np
import subprocess
import os

if os.path.exists('input_read1.wav'):
    os.remove('input_read1.wav')

def animate(i):
    proc_args = ['arecord', '-D', 'plughw:1,0', '-d', '1', '-c1', '-M', '-r', '48000', '-f', 'S32_LE', '-t', 'wav',
                 '-V', 'mono', '-v', 'input_read1.wav']
    rec_proc = subprocess.Popen(proc_args, shell=False, preexec_fn=os.setsid)
    print("startRecordingArecord()> rec_proc pid= " + str(rec_proc.pid))

    # wczytywanie pliku z nagraniem
    if os.path.exists('input_read1.wav'):
        Fs, audio = wavfile.read('input_read1.wav', mmap=False)
    else:
        Fs = 48000
        audio = [0] * Fs

    # filtr antyaliasingowy
    audio = filt.filtr_dol(audio, Fs, 8000, 1024)

    # flitr przeciwkoprzydzwiekowi
    audio = filt.filtr_gor(audio, Fs, 360, 1024)

    # decymacja
    Fs, audio = filt.decymacja(audio, Fs, 6)

    # normalization
    rms = np.sqrt(np.mean(audio ** 2))
    audio1 = audio / rms

    # filtr preemfazy
    audio = filt.preemfaza(audio, 0.95)

    audiofft = fft(audio)
    audiofft = (2 / Fs) * np.abs(audiofft[:int(Fs) // 2])
    '''
    ##Filtracja dzwieku wentylatora
    # zapisywanie widma sygnalu
    # if os.path.exists("widmo.csv"):
    #   os.remove("widmo.csv")
    # audiofft.tofile("widmo.csv", sep=';', format='%10.5f')

    # wczytywanie widma sygnalu
    audiox = np.fromfile("widmo.csv", sep=';')
    # print(np.info(audiox))

    for cnt in range(260,300):
        print(np.argmax(audiox[260:300]), np.argmax(audiofft[260:300]))
        if np.argmax(audiox[260:300]) == np.argmax(audiofft[260:300]):
            audiofft -= audiox
            print("Tniemy")
            break

        else:
            audiofft = np.append(audiofft[1:], 0)
            print("NIE Tniemy")
            continue
    '''
    # freq axis
    x_values = np.arange(0, len(audiofft), 1)
    plt.cla()
    plt.plot(x_values, audiofft, label='Signal')
    plt.legend(loc='upper left')
    plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.tight_layout()
plt.show()