from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy.fftpack import fft
import scipy.signal
import numpy as np
import subprocess
import os

# globalne tablice do wyswietlania zlaczonych sygnalow
wholerun1 = [0] * 40000
wholerun2 = [0] * 40000
# globalna wartosc rms do normalizacji
rms1 = 0

if __name__ == '__main__':
    # wyswietlanie ciglego sygnalu za pomoca aniamcji
    def animate(i):
        # nagrywanie 1000 ms sygnalu do pliku
        proc_args = ['arecord', '-D', 'plughw:1,0', '-d', '1', '-c1', '-M', '-r', '48000', '-f', 'S32_LE', '-t', 'wav',
                     '-V', 'mono', '-v', 'input_read1.wav']
        rec_proc = subprocess.Popen(proc_args, shell=False, preexec_fn=os.setsid)
        print("startRecordingArecord()> rec_proc pid= " + str(rec_proc.pid))

        # wczytywanie pliku z nagraniem
        Fs, audio = wavfile.read('input_read1.wav', mmap=True)

        # filtr antyaliasingowy
        filtr = scipy.signal.firwin2(1024, [0, 0.167, 0.183, 1], [1, 1, 0, 0])
        audio = scipy.signal.convolve(audio, filtr, mode='full', method='auto')
        audio = audio[:Fs]

        # flitr przeciwkoprzydzwiekowi
        filtr = scipy.signal.firwin(1023, 160, fs=Fs, pass_zero=False)
        audio = scipy.signal.convolve(audio, filtr, mode='full', method='auto')
        audio = audio[:Fs]

        # decymacja
        audio1 = audio[0::6]
        Fs /= 6

         # normalizacja do rms sygnalu
        rms = np.sqrt(np.mean(audio1 ** 2))
        global rms1
        rms1 = 0.8 * rms1 + 0.2 * rms
        print("Wartosc rms do normalizacji: ", rms1)
        audio1 = audio1 / rms1
        # filtr preemfazy
        audio2 = np.append(audio1[0], audio1[1:] - 0.95 * audio1[:-1])

        # transformata sygnalu
        audiofft = fft(audio2)
        audiofft = (2 / Fs) * np.abs(audiofft[:int(Fs) // 2])

        # prog mocy calego sygnalu (wyznaczany empirycznie)
        prog = 3

        # dlugosc okna w ms * 1000 / Fs
        winlen = 10 * 8
        # wektor mocy sygnalu
        pow_vec = np.ones(len(audio2))

        # petla obliczenia mocy sygnalu w okanach
        for cnt in range(0, len(audio2), winlen):
            tempsamp = audio2[cnt:cnt + winlen]
            pow_vec[cnt:cnt + winlen] *= np.mean(np.abs(fft(tempsamp)))

        # wektor wyroznienia sygnalu z informacja glosowa
        wektor_zazn = np.zeros(len(audio2))
        stan_wysoki = 2

        # wyroznienie fragmentow sygnalu ktorych moc widmowa przekracza wyznaczony prog
        for cnt in range(0, len(wektor_zazn)):
            if pow_vec[cnt] > prog:
                wektor_zazn[cnt] = stan_wysoki

        # licznik aktywnosci sygnalu
        znak = 0
        # petla wyciecia impulsow krotszych niz 100 ms ktore nie sasiaduja z zadnym innym sygnalem
        for cnt in range(0, len(wektor_zazn)):
            if stan_wysoki == wektor_zazn[cnt] and stan_wysoki != wektor_zazn[cnt - 1]:
                znak += 1
            elif stan_wysoki == wektor_zazn[cnt] and znak > 0:
                znak += 1
            # if all(wektor_zazn[cnt:cnt+100*8]) != 0:
            elif stan_wysoki == wektor_zazn[cnt - 1] and stan_wysoki != wektor_zazn[cnt]:
                if znak < 8 * 100 and not any(wektor_zazn[cnt + 1:cnt + 8 * 200 + 1]) and not any(
                        wektor_zazn[cnt - znak - 8 * 200:cnt - znak - 1]):
                    wektor_zazn[cnt - znak:cnt] = 0
                znak = 0

        # petla zaznaczenia 200 ms aktywnosci przed i po sygnale
        cnt = 0
        while cnt < len(wektor_zazn):
            if stan_wysoki == wektor_zazn[cnt] and stan_wysoki != wektor_zazn[cnt - 1] and cnt > 200 * 8:
                # print("\nOperacja 1")
                wektor_zazn[cnt - 200 * 8:cnt] = stan_wysoki
            elif stan_wysoki == wektor_zazn[cnt] and stan_wysoki != wektor_zazn[cnt - 1]:
                # print("\nOperacja 2")
                wektor_zazn[0:cnt] = stan_wysoki
            elif stan_wysoki == wektor_zazn[cnt - 1] and stan_wysoki != wektor_zazn[cnt] and len(audio2) - cnt > 200 * 8:
                # print("\nOperacja 3")
                wektor_zazn[cnt:cnt + 200 * 8] = stan_wysoki
                cnt += 200 * 8 + 2
            elif stan_wysoki == wektor_zazn[cnt - 1] and stan_wysoki != wektor_zazn[cnt]:
                # print("\nOperacja 4")
                wektor_zazn[cnt:len(audio2)] = stan_wysoki
                cnt += 200 * 8
            cnt += 1

        # sklejanie sygnalow
        global wholerun1
        wholerun1 = wholerun1[8000:]
        wholerun1 = np.append(wholerun1, audio2)

        global wholerun2
        wholerun2 = wholerun2[8000:]
        wholerun2 = np.append(wholerun2, wektor_zazn)

        # os czasu
        x_values = np.arange(0, len(wholerun1), 1) / float(Fs)
        # przeskalowanie do sekund
        x_values *= 1000

        plt.cla()
        plt.plot(x_values, wholerun1, x_values, wholerun2, label='Signal')
        plt.legend(loc='upper left')
        plt.tight_layout()

    # aktywacja animacji wyswietlenia sygnalu
    ani = FuncAnimation(plt.gcf(), animate, interval=1000)
    plt.tight_layout()
    plt.show()