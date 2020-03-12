import os
import argparse
import warnings
import numpy as np
from scipy.io import wavfile
from hmmlearn import hmm
import mfcc_lib as fts
import time


# parse input arguments
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Trains the HMM classifier')
    parser.add_argument('--input-folder', dest='input_folder', required=True,
                        help="Input folder with the audio files in subfolders")
    return parser


# define class
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

    start_time = time.time()
    # args = build_arg_parser().parse_args()
    # input_folder = args.input_folder
    input_folder = 'Baza/'

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
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
            # read the input file
            filepath = os.path.join(subfolder, filename)
            Fs, audio = wavfile.read(filepath)

            # extract mfcc features
            mfcc_feats = fts.mfcc(audio, Fs)

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

    # test files
    input_files = [
        'Baza/STOP/008.wav',
        'Baza/W_PRAWO/008.wav',
        'Baza/SZYBCIEJ/008.wav',
        'Baza/WOLNIEJ/008.wav',
        'Baza/W_LEWO/008.wav',
        'Baza/LEKKO/008.wav',
        'Baza/WOZEK/008.wav',
        'Baza/WYLACZ_SIE/008.wav',
        'Baza/WLACZ_SIE/008.wav',
        'Baza/DO_PRZODU/008.wav',
        'Baza/DO_TYLU/008.wav',

    ]

    for input_files in input_files:
        # read file
        Fs, audio = wavfile.read(input_files)

        # extract mfcc features
        mfcc_feats = fts.mfcc(audio, Fs)

        # define variables
        max_score = [float("-inf")]
        output_label = [float("-inf")]

        # iterate through all hmm models and pick highest score
        for item in hmm_models:
            hmm_model, label = item
            score = hmm_model.get_score(mfcc_feats)
            if score > max_score:
                max_score = score
                output_label = label

        print("\nTrue: ", input_files[input_files.find('/') + 1: input_files.rfind('/')])
        print("Predicted: ", output_label)
        warnings.filterwarnings("ignore")

        print("\nUplyniety czas: %s sek" % (time.time() - start_time))