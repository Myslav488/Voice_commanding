from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from lib_mfcc import mfcc, logfbank
import lib_filter as filt
import lib_vad as vad
import numpy as np
import subprocess
import os

# globalne tablice do wyswietlania zlaczonych sygnalow
wholerun1 = [0] * 24000
wholerun2 = [0] * 24000
wholerun3 = [0] * 24000
wyjscie = np.zeros((1,100))
# globalna wartosc rms do normalizacji
rms1 = 10**8


if __name__ == '__main__':

    if os.path.exists('input_read1.wav'):
        os.remove('input_read1.wav')

    # wyswietlanie ciglego sygnalu za pomoca aniamcji
    def animate(i):
        # nagrywanie 1000 ms sygnalu do pliku
        proc_args = ['arecord', '-D', 'plughw:1,0', '-d', '1', '-c1', '-M', '-r', '48000', '-f', 'S32_LE', '-t', 'wav',
                     '-V', 'mono', '-v', 'input_read1.wav']
        rec_proc = subprocess.Popen(proc_args, shell=False, preexec_fn=os.setsid)
        print("startRecordingArecord()> rec_proc pid= " + str(rec_proc.pid))

        # wczytywanie pliku z nagraniem
        if os.path.exists('input_read1.wav'):
            Fs, audio = wavfile.read('input_read1.wav', mmap=False)
        else:
            Fs = 48000
            audio = [0]*Fs

        # filtr antyaliasingowy
        audio = filt.filtr_dol(audio, Fs, 8000, 1024)

        # flitr przeciwkoprzydzwiekowi
        audio = filt.filtr_gor(audio, Fs, 360, 1024)

        # decymacja
        Fs, audio = filt.decymacja(audio, Fs, 6)

         # normalizacja do rms sygnalu
        rms = np.sqrt(np.mean(audio ** 2))
        global rms1
        rms1 = 0.9 * rms1 + 0.1 * rms
        # print("Wartosc rms do normalizacji: ", rms1)
        audio = audio / rms1
        # filtr preemfazy
        audio = filt.preemfaza(audio, 0.95)

        # prog mocy calego sygnalu (wyznaczany empirycznie)
        prog = 9

        # dlugosc okna w ms * 1000 / Fs
        winlen = 10 * 8

        # wektor mocy sygnalu
        # petla obliczenia mocy sygnalu w okanach
        pow_vec = vad.vec_pow(audio, winlen)

        # wektor wyroznienia sygnalu z informacja glosowa
        stan_wysoki = 2

        # wyroznienie fragmentow sygnalu ktorych moc widmowa przekracza wyznaczony prog
        wektor_zazn = vad.wstepne_zazn(pow_vec, prog, stan_wysoki)

        # petla wyciecia impulsow krotszych niz 100 ms ktore nie sasiaduja z zadnym innym sygnalem
        wektor_zazn = vad.usun_krotkie(wektor_zazn, stan_wysoki, Fs)

        # sklejanie sygnalow kilkusekundowe przebiegi
        global wholerun1
        wholerun1 = wholerun1[len(audio):]
        wholerun1 = np.append(wholerun1, audio)

        global wholerun2
        wholerun2 = wholerun2[len(audio):]
        wholerun2 = np.append(wholerun2, wektor_zazn)

        global wholerun3
        wholerun3 = wholerun3[len(audio):]
        wholerun3 = np.append(wholerun3, pow_vec)

        # petla zaznaczenia 300 ms aktywnosci przed i po sygnale, jesli moc sygnalu przekracza polowe progu
        wholerun2 = vad.warunkowe_zazn(wholerun2, wholerun3, Fs, stan_wysoki,prog, 8000, len(wholerun2)-8000)

        # petla zaznaczenia 200 ms aktywnosci przed i po sygnale
        wholerun2 = vad.dodatkowe_zazn(wholerun2, Fs, stan_wysoki, 8000,  len(wholerun2)-8000)

        # ekstrakcja wykrytego sygnalu mowy
        global wyjscie
        temp_wyj = vad.ekstrakcja(wholerun1, wholerun2, stan_wysoki)
        if (len(temp_wyj)>4000):
            wyjscie = temp_wyj

        ''' mfcc_feat = mfcc((wyjscie), Fs)
        # fbank_feat = logfbank(audio1,sampling_freq)
        mfcc_feat = mfcc_feat.T
        # rysowanie wykresow
        plt.cla()
        plt.imshow(mfcc_feat)'''

        rmss = np.ones((len(wyjscie.T)),) * np.sqrt(np.mean(wyjscie ** 2)) # np.var(wyjscie) #

        rmss2 = np.zeros((len(wyjscie.T)),)
        for cnt in range(0, int(len(wyjscie) / winlen)):
            rmss2[cnt*winlen:(cnt+1)*winlen] = np.sqrt(np.mean(wyjscie[cnt*winlen:(cnt+1)*winlen] ** 2))# np.var(wyjscie[cnt*winlen:(cnt+1)*winlen])

        x_values = np.arange(0, len(wyjscie.T), 1)

        Mr = filt.moment_erowy(wyjscie, Fs, 3, winlen)
        Mr /= max(Mr)
        plt.cla()
        plt.plot(x_values, wyjscie.T, x_values, Mr, label='Signal')
        plt.legend(loc='upper left')
        plt.tight_layout()

    # aktywacja animacji wyswietlenia sygnalu
    ani = FuncAnimation(plt.gcf(), animate, interval=1000)
    plt.tight_layout()
    plt.show()
