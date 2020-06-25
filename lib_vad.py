from scipy.fftpack import fft
import scipy.ndimage.filters as fflt
import numpy as np

# petla obliczenia mocy sygnalu w okanach
def vec_pow(signal, winlen):
    pow_vec = np.ones(len(signal))
    for cnt in range(0, len(signal), winlen):
        tempsamp = signal[cnt:cnt + winlen]
        pow_vec[cnt:cnt + winlen] *= np.mean(np.abs(fft(tempsamp)))

    pow_vec = fflt.median_filter(pow_vec, size=9 * winlen)
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
    leng = 100 * coef
    #jak blisko nich nie moga sie znajdowac inne piki(w ms)
    nextto = 300 * coef
    # licznik aktywnosci sygnalu
    g_cnt = 0
    # petla wyciecia impulsow krotszych niz 100 ms ktore nie sasiaduja z zadnym innym sygnalem
    for cnt in range(0, len(vec_sign)):
        if state_high == vec_sign[cnt] and state_high != vec_sign[cnt - 1]:
            g_cnt += 1
        elif state_high == vec_sign[cnt] and g_cnt > 0:
            g_cnt += 1
        elif state_high == vec_sign[cnt - 1] and state_high != vec_sign[cnt]:
            if g_cnt < leng and not any(vec_sign[cnt + 1:cnt + nextto + 1]) and not any(
                    vec_sign[cnt - g_cnt - nextto:cnt - g_cnt - 1]):
                vec_sign[cnt - g_cnt:cnt] = 0
            g_cnt = 0

    return vec_sign


# funkcja zaznaczenia 300 ms aktywnosci przed i po sygnale,
# jesli moc fragmentu sygnalu przekracza polowe progu.
def cond_sign(vec_sign, pow_vec, Fs, state_high=1, thres=1, begin=0, end=-1):
    # wsp. przeliczenia podanego czasu w ms na ilosc probek
    coef = int(Fs/1000)
    # ile ms sygnalu zaznczayc
    leng = 400 * coef
    cnt = begin
    # petla wykrywajaca poczatek i koniec zaznaczenia i zaznaczjaca po nich jesli moc przekracza polowe progu
    while cnt < end:
        if state_high == vec_sign[cnt] and state_high != vec_sign[cnt - 1] and any(
                thres / 2 < pow_vec[cnt - leng :cnt]):
            vec_sign[cnt - leng:cnt] = state_high
        elif state_high == vec_sign[cnt - 1] and state_high != vec_sign[cnt] and any(
                thres / 2 < pow_vec[cnt:cnt + leng]):
            vec_sign[cnt:cnt + leng] = state_high
            cnt += leng -1
        cnt += 1

    return vec_sign

# petla zaznaczenia 100 ms aktywnosci przed i po sygnale
def extra_sign(vec_sign, Fs, state_high=1, begin=0, end=-1):
    # wsp. przeliczenia podanego czasu w ms na ilosc probek
    coef = int(Fs/1000)
    # ile ms sygnalu zaznczayc(w ms)
    leng = 200 * coef
    cnt = begin
    while cnt < end:
        if state_high == vec_sign[cnt] and state_high != vec_sign[cnt - 1] and cnt > leng:
            vec_sign[cnt - leng:cnt] = state_high
        elif state_high == vec_sign[cnt - 1] and state_high != vec_sign[cnt] and len(
                vec_sign) - cnt > leng:
            vec_sign[cnt:cnt + leng] = state_high
            cnt += leng + 2
        cnt += 1
    return vec_sign

def extra2_sign(vec_sign, vec_pow, thres, Fs, state_high=1, begin=0, end=-1):
    assert len(vec_pow) == len(vec_sign), "Vectors should be the same length."

    # wsp. przeliczenia podanego czasu w ms na ilosc probek
    coef = int(Fs / 1000)
    # ile ms sygnalu zaznczayc(w ms)
    leng = 100*coef

    cnt = begin
    while cnt < end:
        # znajdz narastajace zbocze w wektorze indykatorow
        if state_high == vec_sign[cnt] and state_high != vec_sign[cnt - 1] and cnt > leng:
            # stosunek probek powyzej progu do wszystkich w przedziale analizowanym
            xy = len([x for x in vec_pow[cnt-leng:cnt] if x > thres]) / leng
            if (xy > 0.5):
                # jesli wiecej niz polowa probek jest ponizej progu w analizowanym fragmencie, zaznacz go
                vec_sign[cnt-leng:cnt] = state_high
                # stosunek probek powyzej progu do wszystkich w przedziale analizowanym
                xy = len([x for x in vec_pow[cnt - 2*leng:cnt-leng] if x > 2*thres/3]) / leng
                if (xy > 0.5) and cnt > 2*leng:
                    # jesli wiecej niz polowa probek jest ponizej progu w analizowanym fragmencie, zaznacz go
                    vec_sign[cnt - 2*leng:cnt-leng] = state_high
                    # stosunek probek powyzej progu do wszystkich w przedziale analizowanym
                    xy = len([x for x in vec_pow[cnt - 3 * leng:cnt - 2*leng] if x > thres/2]) / leng
                    if (xy > 0.5) and cnt > 3* leng:
                        # jesli wiecej niz polowa probek jest ponizej progu w analizowanym fragmencie, zaznacz go
                        vec_sign[cnt - 3 * leng:cnt - 2*leng] = state_high

        elif state_high == vec_sign[cnt - 1] and state_high != vec_sign[cnt] and len(
                vec_sign) - cnt > leng:
            # stosunek probek powyzej progu do wszystkich w przedziale analizowanym
            xy = len([x for x in vec_pow[cnt:cnt + leng] if x > thres]) / leng
            if (xy > 0.5):
                # jesli wiecej niz polowa probek jest ponizej progu w analizowanym fragmencie, zaznacz go
                vec_sign[cnt:cnt + leng] = state_high
                # stosunek probek powyzej progu do wszystkich w przedziale analizowanym
                xy = len([x for x in vec_pow[cnt+leng:cnt + 2*leng] if x > 2*thres/3]) / leng
                if (xy > 0.5) and len(vec_sign) - cnt > 2*leng:
                    # jesli wiecej niz polowa probek jest ponizej progu w analizowanym fragmencie, zaznacz go
                    vec_sign[cnt+leng:cnt + 2*leng]  = state_high
                    # stosunek probek powyzej progu do wszystkich w przedziale analizowanym
                    xy = len([x for x in vec_pow[cnt+2*leng:cnt + 3*leng] if x > thres/2]) / leng
                    if (xy > 0.5) and len(vec_sign) - cnt > 3*leng:
                        # jesli wiecej niz polowa probek jest ponizej progu w analizowanym fragmencie, zaznacz go
                        vec_sign[cnt+2*leng:cnt + 3*leng] = state_high
                        cnt += (3*leng + 2)
                    cnt += (2 * leng + 2)
                cnt += (leng + 2)
        cnt += 1
    return vec_sign

# ekstrakcja wykrytego sygnalu mowy
def extraction(signal, vec_sign, state_high):
    # tablicaw wycinkow z sygnalu indeksow i dlugosci zaznaczenia
    result = np.zeros((20, 2))
    # licznik iterujacy caly sygnal
    cnt1 = 0
    # licznik iterujacy tablice wynikow
    cnt2 = 0
    while cnt1 < (len(signal) - 16000):
        # wykryj poczatek zaznaczenia
        if state_high == vec_sign[cnt1] and state_high != vec_sign[cnt1 - 1] and cnt1 < 16000:
            result[cnt2, 0] = cnt1
            result[cnt2, 1] += 1
            cnt1 += 1
        elif state_high == vec_sign[cnt1] and state_high == vec_sign[cnt1 - 1] and 12000 < cnt1 < 32000 and result[cnt2, 0] != 0:
            # oblicz dlugosc zaznaczenia
            result[cnt2, 1] += 1
            cnt1 += 1
        elif (state_high != vec_sign[cnt1] and state_high == vec_sign[cnt1 - 1]) or (state_high == vec_sign[cnt1] and cnt1 == 32000-1):
            #wykryj koniec zaznaczenia i inkrementacja licznika 2
            cnt2 += 1
            cnt1 += 1
        else:
            cnt1 += 1

    # wybr najdluzszego odcinka czasowego z tablicy wycinkow
    z = np.argmax(result[:,1])
    print("UWAGA PATRZEC", result[z, :])
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