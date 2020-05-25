from scipy.fftpack import fft
import numpy as np

# petla obliczenia mocy sygnalu w okanach
def vec_pow(sygnal, winlen):
    pow_vec = np.ones(len(sygnal))
    for cnt in range(0, len(sygnal), winlen):
        tempsamp = sygnal[cnt:cnt + winlen]
        pow_vec[cnt:cnt + winlen] *= np.mean(np.abs(fft(tempsamp)))
    return pow_vec

# wyroznienie fragmentow sygnalu ktorych moc widmowa przekracza wyznaczony prog
def wstepne_zazn(pow_vec, prog=1, stan_wysoki=1):
    wektor_zazn = np.zeros(len(pow_vec))
    for cnt in range(0, len(wektor_zazn)):
        if pow_vec[cnt] > prog:
            wektor_zazn[cnt] = stan_wysoki
    return wektor_zazn

#funkcja usuwa krotkie, pojedyncze piki
def usun_krotkie(wektor_zazn, stan_wysoki, Fs):
    # wsp. przeliczenia podanego czasu w ms na ilosc probek
    wsp = int(Fs/1000)
    # jak krotkie(w ms)
    dlg = 200
    #jak blisko nich nie moga sie znajdowac inne piki(w ms)
    obok = 300
    # licznik aktywnosci sygnalu
    znak = 0
    # petla wyciecia impulsow krotszych niz 100 ms ktore nie sasiaduja z zadnym innym sygnalem
    for cnt in range(0, len(wektor_zazn)):
        if stan_wysoki == wektor_zazn[cnt] and stan_wysoki != wektor_zazn[cnt - 1]:
            znak += 1
        elif stan_wysoki == wektor_zazn[cnt] and znak > 0:
            znak += 1
        elif stan_wysoki == wektor_zazn[cnt - 1] and stan_wysoki != wektor_zazn[cnt]:
            if znak < wsp * dlg and not any(wektor_zazn[cnt + 1:cnt + wsp * obok + 1]) and not any(
                    wektor_zazn[cnt - znak - wsp * obok:cnt - znak - 1]):
                wektor_zazn[cnt - znak:cnt] = 0
            znak = 0

    return wektor_zazn


# funkcja zaznaczenia 300 ms aktywnosci przed i po sygnale,
# jesli moc fragmentu sygnalu przekracza polowe progu.
def warunkowe_zazn(wektor_zazn, pow_vec, Fs, stan_wysoki=1, prog=1, poczatek=0, koniec=-1):
    # wsp. przeliczenia podanego czasu w ms na ilosc probek
    wsp = int(Fs/1000)
    # ile ms sygnalu zaznczayc
    ile = 300
    cnt = poczatek
    # petla wykrywajaca poczatek i koniec zaznaczenia i zaznaczjaca po nich jesli moc przekracza polowe progu
    while cnt < koniec:
        if stan_wysoki == wektor_zazn[cnt] and stan_wysoki != wektor_zazn[cnt - 1] and any(
                prog / 2 < pow_vec[cnt - ile * wsp:cnt]):
            wektor_zazn[cnt - ile * wsp:cnt] = stan_wysoki
        elif stan_wysoki == wektor_zazn[cnt - 1] and stan_wysoki != wektor_zazn[cnt] and any(
                prog / 2 < pow_vec[cnt:cnt + ile * wsp]):
            wektor_zazn[cnt:cnt + ile * wsp] = stan_wysoki
            cnt += ile * wsp
        cnt += 1

    return wektor_zazn

# petla zaznaczenia 100 ms aktywnosci przed i po sygnale
def dodatkowe_zazn(wektor_zazn, Fs, stan_wysoki=1,poczatek=0, koniec=-1):
    # wsp. przeliczenia podanego czasu w ms na ilosc probek
    wsp = int(Fs/1000)
    # ile ms sygnalu zaznczayc(w ms)
    ile = 100
    cnt = poczatek
    while cnt < koniec:
        if stan_wysoki == wektor_zazn[cnt] and stan_wysoki != wektor_zazn[cnt - 1] and cnt > ile * wsp:
            wektor_zazn[cnt - ile * wsp:cnt] = stan_wysoki
        elif stan_wysoki == wektor_zazn[cnt - 1] and stan_wysoki != wektor_zazn[cnt] and len(
                wektor_zazn) - cnt > ile * wsp:
            wektor_zazn[cnt:cnt + ile * wsp] = stan_wysoki
            cnt += ile * wsp + 2
        cnt += 1
    return wektor_zazn

# ekstrakcja wykrytego sygnalu mowy
def ekstrakcja(sygnal, wektor_zazn, stan_wysoki):
    # tablicaw wycinkow z sygnalu
    wyniki = np.zeros((20, 2))
    cnt1 = 0
    cnt2 = 0
    while cnt1 < (len(sygnal) - 8000):
        # wykryj poczatek zaznaczenia
        if stan_wysoki == wektor_zazn[cnt1] and stan_wysoki != wektor_zazn[cnt1 - 1] and cnt1 < 8000:
            wyniki[cnt2, 0] = cnt1
            wyniki[cnt2, 1] += 1
            cnt1 += 1
        elif stan_wysoki == wektor_zazn[cnt1] and stan_wysoki == wektor_zazn[cnt1 - 1] and 4000 < cnt1 < 16000:
            # oblicz dlugosc zaznaczenia
            wyniki[cnt2, 1] += 1
            cnt1 += 1
        elif (stan_wysoki != wektor_zazn[cnt1] and stan_wysoki == wektor_zazn[cnt1 - 1]) or (stan_wysoki == wektor_zazn[cnt1] and cnt1 > 16000):
            #wykryj koniec zaznaczenia i inkrementacja licznika 2
            cnt2 += 1
            cnt1 += 1
        else:
            cnt1 += 1

        # # wybr najdluzszego odcinka czasowego z tablicy wycinkow
        z = np.argmax(wyniki[:,1])
    # wytnij z sygnalu fragment od wyznaczonego indeksu
    return sygnal[int(wyniki[z,0]):int(wyniki[z,0])+int(wyniki[z,1])]

def obcinanie_brzegow(fragment, prog):
    #dlugosc brzegu do obciecia(10% fragmentu)
    brzeg = int(0.1*len(fragment))
    #stosunek probek powyzej progu do wszystkich w przedziale poczatkowego brzegu
    xy = len([x for x in fragment[:brzeg] if x < prog ]) / len(fragment[:brzeg])

    # jesli wiecej niz polowa probek jest ponizej progu w brzegowym fragmencie, wytnij go
    if (xy > 0.5):
        fragment = fragment[brzeg:]

    # stosunek probek powyzej progu do wszystkich w przedziale koncowego brzegu
    xy = len([x for x in fragment[-brzeg:] if x < prog]) / len(
        fragment[-brzeg:])

    # jesli wiecej niz polowa probek jest ponizej progu w brzegowym fragmencie, wytnij go
    if (xy > 0.5):
        fragment = fragment[:-brzeg]

    return fragment