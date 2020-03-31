from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from lib_mfcc import mfcc, logfbank
import lib_filter as filt
import lib_vad as vad
from hmmlearn import hmm
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
rms1 = 10**3


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

            # flitr przeciwkoprzydzwiekowi
            audio = filt.filtr_gor(audio, Fs, 360, 1024)

            # normalizacja do rms sygnalu
            rms = np.sqrt(np.mean(audio ** 2))
            audio = audio / rms
            # filtr preemfazy
            audio = filt.preemfaza(audio, 0.95)

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
        proc_args = ['arecord', '-D', 'plughw:1,0', '-d', '1', '-c1', '-M', '-r', '48000', '-f', 'S16_LE', '-t', 'wav',
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
        # print("RMS z maina: ", rms)
        global rms1
        rms1 = 0.8 * rms1 + 0.2 * rms
        # print("Wartosc rms do normalizacji: ", rms1)
        audio = audio / rms1
        # filtr preemfazy
        audio = filt.preemfaza(audio, 0.95)

        # prog mocy calego sygnalu (wyznaczany empirycznie)
        prog = 8

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
        wholerun2 = vad.warunkowe_zazn(wholerun2, wholerun3, Fs, stan_wysoki, 8000, len(wholerun2) - 8000)

        # petla zaznaczenia 200 ms aktywnosci przed i po sygnale
        wholerun2 = vad.dodatkowe_zazn(wholerun2, Fs, stan_wysoki, 8000, len(wholerun2) - 8000)

        # ekstrakcja wykrytego sygnalu mowy
        global wyjscie
        temp_wyj = vad.ekstrakcja(wholerun1, wholerun2, stan_wysoki)
        if (len(temp_wyj) > 4000):
            wyjscie = temp_wyj

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

        print("\nCzas rozpoznawania slowa: %s sek" % (time.time() - g_time))
        g_time = time.time()

        # aktywacja animacji wyswietlenia sygnalu
    ani = FuncAnimation(plt.gcf(), animate, interval=1000)
    plt.tight_layout()
    plt.show()
