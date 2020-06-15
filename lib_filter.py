import scipy.signal
import scipy.fftpack as fft
import numpy as np
import librosa


def filter_lpf(audio, Fs, cutoff, filtrlen=1024):
    # filtr dolnopasmowy
    # design filter
    filtr = scipy.signal.firwin2(filtrlen, [0, cutoff/Fs, 1.1*cutoff/Fs, 1], [1, 1, 0, 0])
    # dokonaj splotu filtru i syganlu
    audio = scipy.signal.convolve(audio, filtr, mode='full', method='auto')
    # wytnij nadmierne fragmenty sygnalu po operacji splotu
    audio = audio[filtrlen//2-1:(-filtrlen//2)]
    # audio = audio[:Fs]
    return audio

def filter_hpf(audio, Fs, cutoff, filtrlen=1024):
    # flitr gornopasmowy
    filtr = scipy.signal.firwin(filtrlen-1, cutoff, fs=Fs, pass_zero=False)
    audio = scipy.signal.convolve(audio, filtr, mode='full', method='auto')
    audio = audio[filtrlen//2-1:(-filtrlen//2)+1]
    return audio

def decimation(audio, Fs, X=1):
    audio = audio[::X]
    Fs /= X
    return Fs, audio

def preemphasis(audio, a=0.95):
    return np.append(audio[0], audio[1:] - a * audio[:-1])

def normalization(audio, x):
    audio /= x
    return audio

def ded_mean(audio, frame):
    val_mean = 0
    for cnt in range(0, int(len(audio) / frame)):
        val_mean = val_mean + (np.mean(audio[cnt * frame:(cnt + 1) * frame])) / frame
    return audio-val_mean

def filter_spectrcut(audio, Fs, frame = 20, Thr = 0.1):
    frame = frame * 1000 / int(Fs)

    audiofft = [0] * int(Fs)
    for cnt1 in np.arange(int(len(audio)/len(audiofft))):
        audiofft = fft.fft(audio[int(cnt1 * frame):int((cnt1 + 1) * frame)])
        audiofft = (2 / Fs) * np.abs(audiofft[:int(Fs) // 2])

        for cnt2 in np.arange(len(audiofft)):
            if audiofft[cnt2] < Thr:
                audiofft[cnt2] = 0

        audio[int(cnt1 * frame):int((cnt1 + 1) * frame)] = fft.ifft(audiofft)

    return audio


''' ###~Nie dziala
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

        M1[int(cnt*ramka):int((cnt+1)*ramka)] = sum([i*x**2 for x, i in enumerate(audiofft)])

    return M1/moment_zerowy(audio, Fs)

def moment_erowy(audio, Fs, r, ramka=15):
    if(0 == r):
        return moment_zerowy(audio, Fs)
    elif(1 == r):
        return moment_pierwszy(audio, Fs, ramka)
    else:
        Mr = np.zeros(len(audio),)
        for cnt in range(0, int(len(audio)/ramka)):
            audiofft = fft.fft(audio[int(cnt*ramka):int((cnt+1)*ramka)])
            audiofft = (2 / Fs) * np.abs(audiofft[:int(Fs//2)])

            M1 = np.mean(moment_pierwszy(audio[int(cnt*ramka):int((cnt+1)*ramka)], Fs, ramka))
            Mr[int(cnt*ramka):int((cnt+1)*ramka)] = sum([((i-M1)**r)*x**2 for x, i in enumerate(audiofft)])
            # print(Mr[int(cnt*ramka):int((cnt+1)*ramka)])

        return Mr/moment_zerowy(audio, Fs)

def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)


def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)

def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)

def removeNoise(audio_clip,noise_clip,n_grad_freq=2,n_grad_time=4,n_fft=2048,
                win_length=2048,hop_length=512,n_std_thresh=1.5,prop_decrease=0.2,):

    # STFT over noise
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # convert to dB
    # Calculate statistics over noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh

    # STFT over signal
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))

    # Calculate value to mask dB to
    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
    # print(noise_thresh, mask_gain_dB)
    # Create a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    # calculate the threshold for each frequency/time bin
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T
    # mask if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh

    # convolve the mask with a smoothing filter
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease

    # mask the signal
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )  # mask real
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    )

    # recover the signal
    recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
    return recovered_signal