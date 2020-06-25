''' # This script allows to supervise parameters of signal and used statistics.'''

from matplotlib.animation import FuncAnimation
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import lib_filter as filt
import lib_vad as vad
import numpy as np
import subprocess
import os
from datetime import datetime

# globalne tablice do wyswietlania zlaczonych sygnalow
g_longsignal = [0] * 24000
g_longsign = [0] * 24000
g_longpower = [0] * 24000

# globalna wartosc rms do normalizacji
g_rms = 0
rmstla = 0

noise = wavfile.read('szum.wav')
noise = noise[::6]
noise = np.transpose(np.array([float((i)) for i in noise]))

if __name__ == '__main__':

    if os.path.exists('record.wav'):
        os.remove('record.wav')

    # wyswietlanie ciglego sygnalu za pomoca aniamcji
    def animate(i):
        # nagrywanie 1000 ms sygnalu do pliku
        proc_args = ['arecord', '-D', 'plughw:1,0', '-d', '1', '-c1', '-M', '-r', '48000', '-f', 'S32_LE', '-t', 'wav',
                     '-V', 'mono', '-v', 'record.wav']
        rec_proc = subprocess.Popen(proc_args, shell=False, preexec_fn=os.setsid)
        print("startRecordingArecord()> rec_proc pid= " + str(rec_proc.pid))

        # wczytywanie pliku z nagraniem
        if os.path.exists('record.wav'):
            try:
                Fs, audio = wavfile.read('record.wav', mmap=False)
            except:
                Fs = 48000
                audio = [0] * Fs

            now = datetime.now()
            dt_string = now.strftime("%d%m%Y_%H%M%S")
            print(dt_string)
        else:
            Fs = 48000
            audio = [0]*Fs

        # filtr antyaliasingowy
        audio = filt.filter_lpf(audio, Fs, 8000, 1024)

        # flitr przeciwkoprzydzwiekowi
        audio = filt.filter_hpf(audio, Fs, 360, 1024)

        # decymacja
        Fs, audio = filt.decimation(audio, Fs, 6)
        audio = np.transpose(np.array([float((i)) for i in audio]))
        audio = filt.removeNoise(audio, noise)
        audio = filt.preemphasis(audio, 0.95)

        # dlugosc okna w ms * 1000 / Fs
        winlen = 10 * 8
        audio = filt.ded_mean(audio, winlen)

        # normalizacja do rms sygnalu
        rms = np.sqrt(np.mean(audio ** 2))

        #utrzymanie rms na stabilnym poziomie
        if rms > 10**8: rms /= (rms // (5*10**7))
        if rms < 5*10**7: rms *= 2
        global g_rms
        if 0 == g_rms:
            g_rms = rms
        if 0 != g_rms:
            g_rms = 0.9 * g_rms + 0.1 * rms
            audio = audio / g_rms

        # filtr preemfazy
        audio = filt.preemphasis(audio, 0.95)

        # audio = filt.filtr_odcinaniezwidma(audio, Fs)

        # prog mocy calego sygnalu (wyznaczany empirycznie)
        thres = 10 # rms/(3*10**6)
        print("Prog to: ", thres)

        # wektor mocy sygnalu
        # petla obliczenia mocy sygnalu w okanach
        vec_pow = vad.vec_pow(audio, winlen)

        # wartosc stanu wysokiego (tylko do wizualizacji danych)
        high_state = 10

        # wyroznienie fragmentow sygnalu ktorych moc widmowa przekracza wyznaczony prog
        vec_sign = vad.presign(vec_pow, thres, high_state)

        # petla wyciecia impulsow krotszych niz 100 ms ktore nie sasiaduja z zadnym innym sygnalem
        vec_sign = vad.rem_short(vec_sign, high_state, Fs)

        # sklejanie sygnalow kilkusekundowe przebiegi
        global g_longsignal
        g_longsignal = g_longsignal[len(audio):]
        g_longsignal = np.append(g_longsignal, audio)

        global g_longsign
        g_longsign = g_longsign[len(audio):]
        g_longsign = np.append(g_longsign, vec_sign)

        global g_longpower
        g_longpower = g_longpower[len(audio):]
        g_longpower = np.append(g_longpower, vec_pow)

        # moc tla
        global rmstla
        # przydziel rmstla wartosci mocy fragmentow sygnalu ktore nie przekraczaja wyznaczonego progu
        for cnt in range(0, int(len(g_longpower) / winlen)):
            if 0 == rmstla and (g_longpower[cnt * winlen]) < thres:
                rmstla = g_longpower[cnt * winlen]
            if 0 != rmstla and (g_longpower[cnt * winlen])*2 < thres: # and wholerun3[cnt * winlen] < 1.5*rmstla
                rmstla = 0.8 * rmstla + 0.2 * g_longpower[cnt * winlen]
        print("Wart skut tla: ", rmstla)

        # funkcja zaznaczenia 300 ms aktywnosci przed i po sygnale, jesli moc sygnalu przekracza polowe progu
        g_longsign = vad.cond_sign(g_longsign, g_longpower, Fs, high_state, thres, 8000, len(g_longsign) - 8000)

        # funkcja zaznaczenia 200 ms aktywnosci przed i po sygnale
        g_longsign = vad.extra_sign(g_longsign, Fs, high_state, 8000, len(g_longsign) - 8000)
        g_longsign = vad.extra2_sign(g_longsign, g_longpower, thres / 2, Fs, high_state, 8000, len(g_longsign) - 8000)

        # ekstrakcja wykrytego sygnalu mowy
        global output
        temp_out = vad.extraction(g_longsignal, g_longsign, high_state)
        if (len(temp_out) > 3000):
            output = temp_out

            if 0 != any(output):
                output = vad.cut_edges(output, thres/2)
                m = np.max(np.abs(output))
                output = (output / m)
                g_longsign[:16000] *= 0

                now = datetime.now()
                dt_string = now.strftime("%d%m%Y_%H%M%S")
                print(dt_string)
                file = "OLDOnes/recs/" + dt_string + ".wav"
                wavfile.write(file, int(Fs), output)

        # os czasu
        x_values = np.arange(0, len(g_longsignal), 1) / float(Fs)
        yrmsv = np.ones((len(g_longsignal),)) * thres # rmstla
        # przeskalowanie osi do sekund
        x_values *= 1000

        # rysowanie wykresow
        plt.cla()
        plt.plot(x_values, g_longsignal, x_values, g_longsign, x_values, g_longpower, x_values, yrmsv)
        plt.tight_layout()

    # aktywacja animacji wyswietlenia sygnalu
    ani = FuncAnimation(plt.gcf(), animate, interval=1000)
    plt.tight_layout()
    plt.show()
