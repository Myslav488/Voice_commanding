#A Library of functions to get signals features for voice recogntions
#made by James Lyons(proper license info at the bottom of the code)
#with my extra modifications

from scipy.fftpack import dct
import decimal
import numpy
import math
import logging


# Calculates the FFT size as a power of two greater than or equal to
def calculate_nfft(samplerate, winlen):
    window_length_samples = winlen * samplerate
    nfft = 1
    while nfft < window_length_samples:
        nfft *= 2
    return nfft

# Compute MFCC features from an audio signal
def mfcc(signal, samplerate=8000, winlen=0.025, winstep=0.01, numcep=13,
         nfilt=26, nfft=None, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True,
         winfunc=lambda x: numpy.hamming((x))):
    nfft = nfft or calculate_nfft(samplerate, winlen)
    feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph, winfunc)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :numcep]
    feat = lifter(feat, ceplifter)
    if appendEnergy: feat[:, 0] = numpy.log(energy)  # replace first cepstral coefficient with log of frame energy
    return feat

# Compute Mel-filterbank energy features from an audio signal.
def fbank(signal, samplerate=8000, winlen=0.025, winstep=0.01,
          nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
          winfunc=lambda x: numpy.hamming((x))):
    highfreq = highfreq or samplerate / 2
    signal = preemphasis(signal, preemph)
    frames = framesig(signal, winlen * samplerate, winstep * samplerate, winfunc)
    pspec = powspec(frames, nfft)
    energy = numpy.sum(pspec, 1)  # this stores the total energy in each frame
    energy = numpy.where(energy == 0, numpy.finfo(float).eps, energy)  # if energy is zero, we get problems with log

    fb = get_filterbanks2(nfilt, nfft, samplerate, lowfreq, highfreq)
    feat = numpy.dot(pspec, fb.T)  # compute the filterbank energies
    feat = numpy.where(feat == 0, numpy.finfo(float).eps, feat)  # if feat is zero, we get problems with log

    return feat, energy


# Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond to fft bins. T
def get_filterbanks(nfilt=20, nfft=512, samplerate=8000, lowfreq=0, highfreq=None):
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = numpy.linspace(lowmel, highmel, nfilt + 2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = numpy.floor((nfft + 1) * mel2hz(melpoints) / samplerate)

    fbank = numpy.zeros([nfilt, nfft // 2 + 1])
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank

    # Compute a Human Factor Mel-filterbank.


def get_filterbanks2(nfilt=20, nfft=1024, samplerate=8000, lowfreq=0, highfreq=None):
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

    # coefficients taken from Makowski's book
    a = 6.23 * 10 ** (-6)
    b = 9.34 * 10 ** (-2)
    c = 28.52
    ejj = lambda x: a * x ** 2 + b * x + c

    ad1 = 0.5 / (700 + lowfreq)
    bd1 = 700 / (700 + lowfreq)
    cd1 = (-0.5 * lowfreq) / (1 + 700 / (700 + lowfreq))

    bk1 = (b - bd1) / (a - ad1)
    ck1 = (c - cd1) / (a - ad1)

    ad2 = -0.5 / (700 + highfreq)
    bd2 = -700 / (700 + highfreq)
    cd2 = 0.5 * highfreq / (1 + 700 / (700 + highfreq))

    bk2 = (b - bd2) / (a - ad2)
    ck2 = (c - cd2) / (a - ad2)

    # compute marginal points as roots of square equation
    fclow = (-bk1 + (bk1 ** 2 - 4 * ck1) ** 0.5) / 2
    fchigh = (-bk2 + (bk2 ** 2 - 4 * ck2) ** 0.5) / 2

    # compute peaks evenly spaced
    melpoints = numpy.linspace((fclow), (fchigh), (nfilt))
    # convert mel to hz
    peakpoints = mel2hz(melpoints)

    # get extreme marginal of filters
    lowpoints = numpy.ones(len(peakpoints)) * (-700 - ejj(peakpoints) + (
                (700 + ejj(peakpoints)) ** 2 + peakpoints * (peakpoints + 1400) - 1400 * ejj(peakpoints)) ** 0.5)
    highpoints = numpy.ones(len(peakpoints)) * (lowpoints + 2 * ejj(peakpoints))

    # normalization
    minn = min(min(lowpoints), min(peakpoints), min(highpoints))
    maxx = max(max(lowpoints), max(peakpoints), max(highpoints))
    peakpoints = numpy.floor((nfft + 1) * (peakpoints - minn) / (2 * maxx))
    lowpoints = numpy.floor((nfft + 1) * (lowpoints - minn) / (2 * maxx))
    highpoints = numpy.floor((nfft + 1) * (highpoints - minn) / (2 * maxx))

    # shaping filters
    fbank = numpy.zeros([nfilt, nfft // 2 + 1])
    for j in range(0, len(peakpoints)):
        for i in range(int(lowpoints[j]), int(peakpoints[j])):
            fbank[j, i] = (i - lowpoints[j]) / (peakpoints[j] - lowpoints[j])
        for i in range(int(peakpoints[j]), int(highpoints[j])):
            fbank[j, i] = (highpoints[j] - i) / (highpoints[j] - peakpoints[j])

    return fbank


# Compute log Mel-filterbank energy features from an audio signal.
def logfbank(signal, samplerate=8000, winlen=0.025, winstep=0.02,
             nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
             winfunc=lambda x: numpy.hamming((x))):
    feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph, winfunc)
    return numpy.log(feat)


def hz2mel(hz):
    return 2595 * numpy.log10(1 + hz / 700.)


def mel2hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def lifter(cepstra, L=22):
    if L > 0:
        nframes, ncoeff = numpy.shape(cepstra)
        n = numpy.arange(ncoeff)
        lift = 1 + (L / 2.) * numpy.sin(numpy.pi * n / L)
        return lift * cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def rolling_window(a, window, step=1):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]


# Frame a signal into overlapping frames.
def framesig(sig, frame_len, frame_step, winfunc=lambda x: numpy.hamming((x)), stride_trick=True):
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = numpy.zeros((padlen - slen,))
    padsignal = numpy.concatenate((sig, zeros))
    if stride_trick:
        win = winfunc(frame_len)
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(
            numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = numpy.array(indices, dtype=numpy.int32)
        frames = padsignal[indices]
        win = numpy.tile(winfunc(frame_len), (numframes, 1))

    return frames * win


# Does overlap-add procedure to undo the action of framesig.
def deframesig(frames, siglen, frame_len, frame_step, winfunc=lambda x: numpy.hamming((x))):
    frame_len = round_half_up(frame_len)
    frame_step = round_half_up(frame_step)
    numframes = numpy.shape(frames)[0]
    assert numpy.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'

    indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(
        numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    indices = numpy.array(indices, dtype=numpy.int32)
    padlen = (numframes - 1) * frame_step + frame_len

    if siglen <= 0: siglen = padlen

    rec_signal = numpy.zeros((padlen,))
    window_correction = numpy.zeros((padlen,))
    win = winfunc(frame_len)

    for i in range(0, numframes):
        window_correction[indices[i, :]] = window_correction[
                                               indices[i, :]] + win + 1e-15  # add a little bit so it is never zero
        rec_signal[indices[i, :]] = rec_signal[indices[i, :]] + frames[i, :]

    rec_signal = rec_signal / window_correction
    return rec_signal[0:siglen]


# Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
def magspec(frames, NFFT):
    if numpy.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            numpy.shape(frames)[1], NFFT)
    complex_spec = numpy.fft.rfft(frames, NFFT)
    return numpy.absolute(complex_spec)


# Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
def powspec(frames, NFFT):
    return 1.0 / NFFT * numpy.square(magspec(frames, NFFT))


# Compute the log power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
def logpowspec(frames, NFFT, norm=1):
    ps = powspec(frames, NFFT);
    ps[ps <= 1e-30] = 1e-30
    lps = 10 * numpy.log10(ps)
    if norm:
        return lps - numpy.max(lps)
    else:
        return lps


def preemphasis(signal, coeff=0.95):
    return numpy.append(signal[0], signal[1:] - coeff * signal[:-1])

'''
The MIT License (MIT)

Copyright (c) 2013 James Lyons

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

>>>Link to original code:
https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/base.py
'''