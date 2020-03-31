import scipy.signal
import numpy as np

def filtr_dol(audio, Fs, cutoff, filtrlen=1024):
    # filtr antyaliasingowy
    filtr = scipy.signal.firwin2(filtrlen, [0, cutoff/Fs, 1.1*cutoff/Fs, 1], [1, 1, 0, 0])
    audio = scipy.signal.convolve(audio, filtr, mode='full', method='auto')
    audio = audio[:Fs]
    return audio

def filtr_gor(audio, Fs, cutoff, filtrlen=1024):
    # flitr przeciwkoprzydzwiekowi
    filtr = scipy.signal.firwin(filtrlen-1, cutoff, fs=Fs, pass_zero=False)
    audio = scipy.signal.convolve(audio, filtr, mode='full', method='auto')
    audio = audio[filtrlen//2-1:(-filtrlen//2)+1]
    return audio

def decymacja(audio, Fs, X=1):
    audio = audio[::X]
    Fs /= X
    return Fs, audio

def preemfaza(audio, a=0.95):
    return np.append(audio[0], audio[1:] - a * audio[:-1])

def normalizacja(audio, x):
    audio /= x
    return audio
