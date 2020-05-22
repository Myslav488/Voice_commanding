import scipy.signal
import scipy.fftpack as fft
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

def odejm_wart_sr(audio, ramka):
    wart_sr = 0
    for cnt in range(0, int(len(audio)/ramka)):
        wart_sr = wart_sr + (np.mean(audio[cnt*ramka:(cnt+1)*ramka]))/ramka
    return audio-wart_sr

def filtr_odcinaniezwidma(audio, Fs, ramka = 20, Thr = 0.1):
    ramka = ramka *1000 / int(Fs)

    audiofft = [0] * int(Fs)
    for cnt1 in np.arange(int(len(audio)/len(audiofft))):
        audiofft = fft.fft(audio[int(cnt1*ramka):int((cnt1+1)*ramka)])
        audiofft = (2 / Fs) * np.abs(audiofft[:int(Fs) // 2])

        for cnt2 in np.arange(len(audiofft)):
            if audiofft[cnt2] < Thr:
                audiofft[cnt2] = 0

        audio[int(cnt1*ramka):int((cnt1+1)*ramka)] = fft.ifft(audiofft)

    return audio


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
def moment_zerowy(audio, Fs):
    audiofft = fft.fft(audio)
    audiofft = (2 / Fs) * np.abs(audiofft[:int(Fs) // 2])
    return sum([ x**2 for x in (audiofft)])

def moment_pierwszy(audio, Fs, ramka=15):
    M1 = np.array(np.arange(len(audio),))
    for cnt in range(0, int(len(audio)/ramka)):
        audiofft = fft.fft(audio[int(cnt*ramka):int((cnt+1)*ramka)])
        audiofft = (2 / Fs) * np.abs(audiofft[:int(Fs//2)])

        M1[int(cnt*ramka):int((cnt+1)*ramka)] =  sum([i*x**2 for x, i in enumerate(audiofft)])

    return M1/moment_zerowy(audio, Fs)

def moment_erowy(audio, Fs, r, ramka=15):
    if(0 == r):
        return moment_zerowy(audio, Fs)
    elif(1 == r):
        return moment_pierwszy(audio, Fs, ramka)
    else:
        Mr = np.array(np.arange(len(audio),))
        for cnt in range(0, int(len(audio)/ramka)):
            audiofft = fft.fft(audio[int(cnt*ramka):int((cnt+1)*ramka)])
            audiofft = (2 / Fs) * np.abs(audiofft[:int(Fs//2)])

            M1 = np.mean(moment_pierwszy(audio[int(cnt*ramka):int((cnt+1)*ramka)], Fs, ramka))
            print(cnt, M1)
            Mr[int(cnt*ramka):int((cnt+1)*ramka)] = sum([((i-M1)**r)*x**2 for x, i in enumerate(audiofft)])

        return Mr/moment_zerowy(audio, Fs)