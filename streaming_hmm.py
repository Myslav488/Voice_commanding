from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy.fftpack import fft
from mfcc_lib import mfcc, logfbank
from hmmlearn import hmm
import scipy.signal
import numpy as np
import subprocess
import warnings
import time
import os

# globalne tablice do wyswietlania zlaczonych sygnalow
wholerun1 = [0] * 24000
wholerun2 = [0] * 24000
wholerun3 = [0] * 24000
wyjscie = np.zeros((1,100))
g_time = time.time()
# globalna wartosc rms do normalizacji
zazn_przd = 0
zazn_tyl = 0
rms1 = 10**8


# definiowanie klasy HMM
class HMMtrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []

        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components, covariance_type=self.cov_type,
                                         n_iter=self.n_iter)

        else:
            raise TypeError('Invalid model type')

    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))

    def get_score(self, input_data):
        return self.model.score(input_data)

if __name__ == '__main__':

    if os.path.exists('input_read1.wav'):
        os.remove('input_read1.wav')

    start_time = time.time()
    input_folder = 'Baza2/'

    hmm_models = []

    for dirname in os.listdir(input_folder):
        subfolder = os.path.join(input_folder, dirname)

        if not os.path.isdir(subfolder):
            continue

        # extracting labels
        label = subfolder[subfolder.rfind('/') + 1:]

        # initialize vars
        X = np.array([])
        y_words = []
        warnings.filterwarnings("ignore")

        # iterating through the audio files,
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')]:
            # read the input file
            filepath = os.path.join(subfolder, filename)
            Fs, audio = wavfile.read(filepath)

            # extract mfcc features
            mfcc_feats = mfcc(audio, Fs)

            if len(X) == 0:
                X = mfcc_feats
            else:
                X = np.append(X, mfcc_feats, axis=0)

            # Append the label
            y_words.append(label)

        # train the hmm model
        hmm_trainer = HMMtrainer()
        hmm_trainer.train(X)
        hmm_models.append((hmm_trainer, label))
        hmm_trainer = None

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
        filtr = scipy.signal.firwin2(1024, [0, 0.167, 0.183, 1], [1, 1, 0, 0])
        audio = scipy.signal.convolve(audio, filtr, mode='full', method='auto')
        audio = audio[:Fs]

        # flitr przeciwkoprzydzwiekowi
        filtr = scipy.signal.firwin(1023, 360, fs=Fs, pass_zero=False)
        audio = scipy.signal.convolve(audio, filtr, mode='full', method='auto')
        audio = audio[:Fs]

        # decymacja
        audio1 = audio[0::6]
        Fs /= 6

         # normalizacja do rms sygnalu
        rms = np.sqrt(np.mean(audio1 ** 2))
        global rms1
        rms1 = 0.8 * rms1 + 0.2 * rms
        # print("Wartosc rms do normalizacji: ", rms1)
        audio1 = audio1 / rms1
        # filtr preemfazy
        audio2 = np.append(audio1[160], audio1[161:] - 0.85 * audio1[160:-1])

        # prog mocy calego sygnalu (wyznaczany empirycznie)
        prog = 6

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
                if znak < 8 * 100 and not any(wektor_zazn[cnt + 1:cnt + 8 * 400 + 1]) and not any(
                        wektor_zazn[cnt - znak - 8 * 400:cnt - znak - 1]):
                    wektor_zazn[cnt - znak:cnt] = 0
                znak = 0

        # sklejanie sygnalow kilkusekundowe przebiegi
        global wholerun1
        wholerun1 = wholerun1[len(audio2):]
        wholerun1 = np.append(wholerun1, audio2)

        global wholerun2
        wholerun2 = wholerun2[len(audio2):]
        wholerun2 = np.append(wholerun2, wektor_zazn)

        global wholerun3
        wholerun3 = wholerun3[len(audio2):]
        wholerun3 = np.append(wholerun3, pow_vec)

        # petla zaznaczenia 300 ms aktywnosci przed i po sygnale, jesli moc sygnalu przekracza polowe progu
        cnt = 8000
        while cnt < len(wholerun2) - 8000:
            if stan_wysoki == wholerun2[cnt] and stan_wysoki != wholerun2[cnt - 1] and any(
                    stan_wysoki / 2 < wholerun3[cnt - 300 * 8:cnt]):
                wholerun2[cnt - 300 * 8:cnt] = stan_wysoki
            elif stan_wysoki == wholerun2[cnt - 1] and stan_wysoki != wholerun2[cnt] and any(
                    stan_wysoki / 2 < wholerun3[cnt:cnt + 300 * 8]):
                wholerun2[cnt:cnt + 300 * 8] = stan_wysoki
                cnt += 300 * 8
            cnt += 1

        # petla zaznaczenia 200 ms aktywnosci przed i po sygnale
        cnt = 8000
        while cnt < len(wholerun2)-8000:
            if stan_wysoki == wholerun2[cnt] and stan_wysoki != wholerun2[cnt - 1] and cnt > 200 * 8:
                wholerun2[cnt - 200 * 8:cnt] = stan_wysoki
            elif stan_wysoki == wholerun2[cnt - 1] and stan_wysoki != wholerun2[cnt] and len(
                    wholerun1) - cnt > 300 * 8:
                wholerun2[cnt:cnt + 200 * 8] = stan_wysoki
                cnt += 200 * 8 + 2
            cnt += 1

        # ekstrakcja wykrytego sygnalu mowy
        wyniki = np.zeros((20, 2))
        cnt1 = 0
        cnt2 = 0
        while cnt1 < (len(wholerun1) - 8000):
            if stan_wysoki == wholerun2[cnt1] and stan_wysoki != wholerun2[cnt1 - 1] and cnt1 < 8000:
                wyniki[cnt2, 0] = cnt1
                wyniki[cnt2, 1] += 1
                cnt1 += 1
            elif stan_wysoki == wholerun2[cnt1] and stan_wysoki == wholerun2[cnt1 - 1] and 4000 < cnt1 < 16000:
                wyniki[cnt2, 1] += 1
                cnt1 += 1
            elif stan_wysoki != wholerun2[cnt1] and stan_wysoki == wholerun2[cnt1 - 1]:
                cnt2 += 1
                cnt1 += 1
            else:
                cnt1 += 1

        z = np.argmax(wyniki[:,1])
        global wyjscie
        if (wyniki[z,1]>4000):
            wyjscie = wholerun1[int(wyniki[z,0]):int(wyniki[z,0])+int(wyniki[z,1])]

        mfcc_feat = mfcc((wyjscie), Fs)
        # mfcc_feat = mfcc_feat.T

        # rysowanie wykresow
        plt.cla()
        plt.imshow(mfcc_feat.T)

        # define variables
        max_score = [float("-inf")]
        output_label = [float("-inf")]

        # iterate through all hmm models and pick highest score
        for item in hmm_models:
            hmm_model, label = item
            score = hmm_model.get_score(mfcc_feat)
            if score > max_score:
                max_score = score
                output_label = label
            print(label, score)
        print("Predicted: ", output_label)
        warnings.filterwarnings("ignore")
        global g_time

        print("\nUplyniety czas: %s sek" % (time.time() - g_time))
        g_time = time.time()

        # aktywacja animacji wyswietlenia sygnalu
    ani = FuncAnimation(plt.gcf(), animate, interval=1000)
    plt.tight_layout()
    plt.show()
