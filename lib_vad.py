from scipy.fftpack import fft
import numpy as np

# petla obliczenia mocy sygnalu w okanach
def vec_pow(signal, winlen):
    pow_vec = np.ones(len(signal))
    for cnt in range(0, len(signal), winlen):
        tempsamp = signal[cnt:cnt + winlen]
        pow_vec[cnt:cnt + winlen] *= np.mean(np.abs(fft(tempsamp)))
    return pow_vec

# wyroznienie fragmentow sygnalu ktorych moc widmowa przekracza wyznaczony prog
def presign(pow_vec, thres=1, state_high=1):
    vec_sign = np.zeros(len(pow_vec))
    for cnt in range(0, len(vec_sign)):
        if pow_vec[cnt] > thres:
            vec_sign[cnt] = state_high
    return vec_sign

#funkcja usuwa krotkie, pojedyncze piki
def rem_short(vec_sign, state_high, Fs):
    # wsp. przeliczenia podanego czasu w ms na ilosc probek
    coef = int(Fs/1000)
    # jak krotkie(w ms)
    leng = 200
    #jak blisko nich nie moga sie znajdowac inne piki(w ms)
    nextto = 300
    # licznik aktywnosci sygnalu
    g_cnt = 0
    # petla wyciecia impulsow krotszych niz 100 ms ktore nie sasiaduja z zadnym innym sygnalem
    for cnt in range(0, len(vec_sign)):
        if state_high == vec_sign[cnt] and state_high != vec_sign[cnt - 1]:
            g_cnt += 1
        elif state_high == vec_sign[cnt] and g_cnt > 0:
            g_cnt += 1
        elif state_high == vec_sign[cnt - 1] and state_high != vec_sign[cnt]:
            if g_cnt < coef * leng and not any(vec_sign[cnt + 1:cnt + coef * nextto + 1]) and not any(
                    vec_sign[cnt - g_cnt - coef * nextto:cnt - g_cnt - 1]):
                vec_sign[cnt - g_cnt:cnt] = 0
            g_cnt = 0

    return vec_sign


# funkcja zaznaczenia 300 ms aktywnosci przed i po sygnale,
# jesli moc fragmentu sygnalu przekracza polowe progu.
def cond_sign(vec_sign, pow_vec, Fs, state_high=1, thres=1, begin=0, end=-1):
    # wsp. przeliczenia podanego czasu w ms na ilosc probek
    coef = int(Fs/1000)
    # ile ms sygnalu zaznczayc
    leng = 300
    cnt = begin
    # petla wykrywajaca poczatek i koniec zaznaczenia i zaznaczjaca po nich jesli moc przekracza polowe progu
    while cnt < end:
        if state_high == vec_sign[cnt] and state_high != vec_sign[cnt - 1] and any(
                thres / 2 < pow_vec[cnt - leng * coef:cnt]):
            vec_sign[cnt - leng * coef:cnt] = state_high
        elif state_high == vec_sign[cnt - 1] and state_high != vec_sign[cnt] and any(
                thres / 2 < pow_vec[cnt:cnt + leng * coef]):
            vec_sign[cnt:cnt + leng * coef] = state_high
            cnt += leng * coef
        cnt += 1

    return vec_sign

# petla zaznaczenia 100 ms aktywnosci przed i po sygnale
def extra_sign(vec_sign, Fs, state_high=1, begin=0, end=-1):
    # wsp. przeliczenia podanego czasu w ms na ilosc probek
    coef = int(Fs/1000)
    # ile ms sygnalu zaznczayc(w ms)
    leng = 100
    cnt = begin
    while cnt < end:
        if state_high == vec_sign[cnt] and state_high != vec_sign[cnt - 1] and cnt > leng * coef:
            vec_sign[cnt - leng * coef:cnt] = state_high
        elif state_high == vec_sign[cnt - 1] and state_high != vec_sign[cnt] and len(
                vec_sign) - cnt > leng * coef:
            vec_sign[cnt:cnt + leng * coef] = state_high
            cnt += leng * coef + 2
        cnt += 1
    return vec_sign

# ekstrakcja wykrytego sygnalu mowy
def extraction(signal, vec_sign, state_high):
    # tablicaw wycinkow z sygnalu
    result = np.zeros((20, 2))
    cnt1 = 0
    cnt2 = 0
    while cnt1 < (len(signal) - 8000):
        # wykryj poczatek zaznaczenia
        if state_high == vec_sign[cnt1] and state_high != vec_sign[cnt1 - 1] and cnt1 < 8000:
            result[cnt2, 0] = cnt1
            result[cnt2, 1] += 1
            cnt1 += 1
        elif state_high == vec_sign[cnt1] and state_high == vec_sign[cnt1 - 1] and 4000 < cnt1 < 16000:
            # oblicz dlugosc zaznaczenia
            result[cnt2, 1] += 1
            cnt1 += 1
        elif (state_high != vec_sign[cnt1] and state_high == vec_sign[cnt1 - 1]) or (state_high == vec_sign[cnt1] and cnt1 > 16000):
            #wykryj koniec zaznaczenia i inkrementacja licznika 2
            cnt2 += 1
            cnt1 += 1
        else:
            cnt1 += 1

        # # wybr najdluzszego odcinka czasowego z tablicy wycinkow
        z = np.argmax(result[:,1])
    # wytnij z sygnalu fragment od wyznaczonego indeksu
    return signal[int(result[z, 0]):int(result[z, 0]) + int(result[z, 1])]

def cut_edges(extract, thres):
    #dlugosc brzegu do obciecia(10% fragmentu)
    edge = int(0.1 * len(extract))
    #stosunek probek powyzej progu do wszystkich w przedziale poczatkowego brzegu
    xy = len([x for x in extract[:edge] if x < thres]) / len(extract[:edge])

    # jesli wiecej niz polowa probek jest ponizej progu w brzegowym fragmencie, wytnij go
    if (xy > 0.5):
        extract = extract[edge:]

    # stosunek probek powyzej progu do wszystkich w przedziale koncowego brzegu
    xy = len([x for x in extract[-edge:] if x < thres]) / len(
        extract[-edge:])

    # jesli wiecej niz polowa probek jest ponizej progu w brzegowym fragmencie, wytnij go
    if (xy > 0.5):
        extract = extract[:-edge]

    return extract