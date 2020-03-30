import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import numpy as np
import time


if __name__ == '__main__':

    Fs, audio = wavfile.read('probka.wav', mmap=True)
    start_time = time.time()

    # audio = audio[30000:38000]

    # transformata sygnalu
    audiofft = fft(audio)
    audiofft = (2 / Fs) * np.abs(audiofft[:int(Fs) // 2])
    #wektor czest. do rysowania widma
    freq = np.arange(0, len(audiofft), 1)
    # moc calego sygnalu
    prog = 1000

    # dlugosc okna w ms * 1000 / Fs
    winlen = 10 * 8
    # wektory cech sygnalu do wyswietlania
    pow_vec = np.ones(len(audio))

    # petla obliczen
    for cnt in range(0, len(audio), winlen):
        tempsamp = audio[cnt:cnt+winlen]
        pow_vec[cnt:cnt+winlen] *=  np.mean(np.abs(fft(tempsamp)))

    # time axis
    czas = np.arange(0, len(audio), 1) / float(Fs)
    plt.figure(1)
    plt.plot(czas, audio, czas, pow_vec)

    wektor_zazn = np.zeros(len(audio))

    for cnt in range(0, len(audio)):
        if pow_vec[cnt] > prog:
            wektor_zazn[cnt] = 5000

    print("\nUplyniety czas: %s sek" % (time.time() - start_time))

    plt.figure(2)
    plt.plot(czas, audio, czas, wektor_zazn)
    znak = 0
    for cnt in range(0, len(audio)):
        if 5000 == wektor_zazn[cnt] and 5000 != wektor_zazn[cnt - 1]:
            znak += 1
        elif 5000 == wektor_zazn[cnt] and znak > 0:
            znak += 1
        #if all(wektor_zazn[cnt:cnt+100*8]) != 0:
        elif 5000 == wektor_zazn[cnt - 1] and 5000 != wektor_zazn[cnt]:
            if znak < 8*100 and not any(wektor_zazn[cnt+1:cnt+8*200+1]) and not any(wektor_zazn[cnt-znak-8*200:cnt-znak-1]):
                wektor_zazn[cnt-znak:cnt] = 0
            znak = 0


    plt.figure(3)
    plt.plot(czas, audio, czas, wektor_zazn)

    cnt = 0
    while cnt < len(audio):
        if 5000 == wektor_zazn[cnt] and 5000 != wektor_zazn[cnt-1] and cnt > 200*8:
            print("\nOperacja 1")
            wektor_zazn[cnt-200*8:cnt] = 5000
        elif 5000 == wektor_zazn[cnt] and 5000 != wektor_zazn[cnt-1]:
            print("\nOperacja 2")
            wektor_zazn[0:cnt] = 5000
        elif 5000 == wektor_zazn[cnt-1] and 5000 != wektor_zazn[cnt] and len(audio)-cnt > 200*8:
            print("\nOperacja 3")
            wektor_zazn[cnt:cnt+200*8] = 5000
            cnt += 200*8+2
        elif 5000 == wektor_zazn[cnt-1] and 5000 != wektor_zazn[cnt]:
            print("\nOperacja 4")
            wektor_zazn[cnt:len(audio)] = 5000
            cnt += 200*8
        cnt +=1

    print("\nUplyniety czas: %s sek" % (time.time() - start_time))

    plt.figure(4)
    plt.plot(czas, audio, czas, wektor_zazn)


    plt.show()

