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
from datetime import datetime
import pickle

# globalne tablice do wyswietlania zlaczonych sygnalow
g_longsignal = [0] * 24000
g_longsign = [0] * 24000
g_longpower = [0] * 24000
output = np.zeros((1, 100))
g_time = time.time()
ster_time = time.time()
# globalna wartosc rms do normalizacji
g_rms = 0
rmstla = 0

noise = wavfile.read('szum.wav')
noise = noise[::6]
noise = np.transpose(np.array([float((i)) for i in noise]))

# definiowanie klasy HMM
class HMMtrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=15, cov_type='diag', n_iter=200):
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

    if os.path.exists('record.wav'):
        os.remove('record.wav')

    start_time = time.time()
    input_folder = 'Baza2/'

    hmm_models = []

    if os.path.exists('model_HMM.pkl'):
        hmm_models = pickle.load(open('model_HMM.pkl', "rb"))
    else:
        # budowanie modelu klasyfikatora HMM
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
                audio = filt.filter_hpf(audio, Fs, 360, 1024)

                audio = np.transpose(np.array([float((i)) for i in audio]))
                audio = filt.removeNoise(audio, noise)

                # normalizacja do rms sygnalu
                rms = np.sqrt(np.mean(audio ** 2))
                # print("RMS uczonych probek ",rms)
                audio = audio / rms
                # filtr preemfazy
                audio = filt.preemphasis(audio, 0.95)

                # extract mfcc features
                m = np.max(np.abs(audio))
                audio = (audio / m)
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
        pickle.dump(hmm_models, open('model_HMM.pkl', "wb"))
        print("Utworzono model HMM. Czas: ",   ster_time - time.time())
        ster_time = time.time()

    # wyswietlanie ciglego sygnalu za pomoca aniamcji
    def animate(i):
        # nagrywanie 1000 ms sygnalu do pliku
        proc_args = ['arecord', '-D', 'plughw:1,0', '-d', '1', '-c1', '-M', '-r', '48000', '-f', 'S32_LE', '-t', 'wav',
                     '-V', 'mono', '-v', 'record.wav']
        rec_proc = subprocess.Popen(proc_args, shell=False, preexec_fn=os.setsid)
        # print("startRecordingArecord()> rec_proc pid= " + str(rec_proc.pid))
        global ster_time
        print("Cos tam: ", time.time() - ster_time)
        ster_time = time.time()
        # wczytywanie pliku z nagraniem
        if os.path.exists('record.wav'):
            try:
                Fs, audio = wavfile.read('record.wav', mmap=False)
            except:
                Fs = 48000
                audio = [0] * Fs
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
        # odejmowanie wart sredniej
        audio = filt.ded_mean(audio, winlen)

        # normalizacja do rms sygnalu
        rms = np.sqrt(np.mean(audio ** 2))

        # utrzymanie rms na stabilnym poziomie
        if rms > 10**8: rms /= (rms//(5*10**7))
        if rms < 5*10**7: rms *= 2
        global g_rms
        if 0 == g_rms:
            g_rms = rms
        if 0 != g_rms:
            g_rms = 0.9*g_rms + 0.1*rms
            audio = audio / g_rms
        # filtr preemfazy
        # audio = filt.preemfaza(audio, 0.95)
        print("Wstepne przetwarzanie: ", time.time() - ster_time)
        ster_time = time.time()
        # prog mocy calego sygnalu (wyznaczany empirycznie)
        thres = rms/(4*10**6)
        # print("RMS to: ", rms, "PRoG ", thres)

        # wektor mocy sygnalu
        # petla obliczenia mocy sygnalu w okanach
        vec_pow = vad.vec_pow(audio, winlen)

        # wektor wyroznienia sygnalu z informacja glosowa
        high_state = 10

        # wyroznienie fragmentow sygnalu ktorych moc widmowa przekracza wyznaczony prog
        vec_sign = vad.presign(vec_pow, thres, high_state)

        # funkcja wyciecia impulsow krotszych niz 100 ms ktore nie sasiaduja z zadnym innym sygnalem
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
            if 0 != rmstla and (g_longpower[cnt * winlen]) * 2 < thres:  # and wholerun3[cnt * winlen] < 1.5*rmstla
                rmstla = 0.2 * rmstla + 0.8 * g_longpower[cnt * winlen]
        print("Wart skut tla: ", rmstla)

        # print("RMS tla to ", rmstla)
        # petla zaznaczenia 300 ms aktywnosci przed i po sygnale, jesli moc sygnalu przekracza polowe progu
        g_longsign = vad.cond_sign(g_longsign, g_longpower, Fs, high_state, thres, 8000, len(g_longsign) - 8000)

        # petla zaznaczenia 200 ms aktywnosci przed i po sygnale
        g_longsign = vad.extra_sign(g_longsign, Fs, high_state, 8000, len(g_longsign) - 8000)
        g_longsign = vad.extra2_sign(g_longsign, g_longpower, thres / 2, Fs, high_state, 8000, len(g_longsign) - 8000)

        # ekstrakcja wykrytego sygnalu mowy
        global output
        temp_out= vad.extraction(g_longsignal, g_longsign, high_state)

        # jesli wyekrahowana probka jest dluzsza niz 375 ms nadpisz ja jako wykryta komende
        if (len(temp_out) > 3000):
            output = temp_out # vad.cut_edges(temp_out, rmstla)

            if 0 != any(output):
                output = vad.cut_edges(output, rmstla)
                m = np.max(np.abs(output))
                output = (output / m)
                g_longsign[:16000] *= 0

            '''now = datetime.now()
            dt_string = now.strftime("%d%m%Y_%H%M%S")
            file = "OLDOnes/recs/" + dt_string + ".wav"
            wavfile.write(file, int(Fs), output)'''
        print("Ekstrakcja: ", time.time() - ster_time)
        ster_time = time.time()
        # print((len(output)/8000))
        mfcc_feat = mfcc((output), Fs)
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
        print("Samo rozpoznawanie: ", time.time() - ster_time)
        ster_time = time.time()
        print("\nCzas rozpoznawania slowa: %s sek" % (time.time() - g_time))
        g_time = time.time()

        # aktywacja animacji wyswietlenia sygnalu
    ani = FuncAnimation(plt.gcf(), animate, interval=1000)
    plt.tight_layout()
    plt.show()
