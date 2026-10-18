'''
Модуль аналітики / аналізу даних
'''

import numpy as np
import math as mt
import matplotlib.pyplot as plt



def Stat_characteristics_in(SL, Text, filename):
    '''
    статистичні характеристики вибірки з урахуванням тренду
    '''

    Yout = MNK_Stat_characteristics(SL)
    iter = len(Yout)
    SL0 = np.zeros((iter))

    for i in range(iter):
        SL0[i] = SL[i] - Yout[i, 0]

    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)

    print('------------', Text,'-------------')
    print('кількість елементів вбірки=', iter)
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('-----------------------------------------------------')

    plt.hist(SL0, bins=20, facecolor="blue", alpha=0.5)
    plt.savefig(filename)
    plt.show()

    return



def MNK_Stat_characteristics(S0):
    '''
    МНК згладжуваннядля визначення стат. характеристик
    '''

    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))

    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)

    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)

    return Yout



def Plot_AV(S0_L, SV_L, Text, filename):
    '''
    графіки тренда, вимірів з шумом
    '''

    plt.clf()
    plt.plot(SV_L)
    plt.plot(S0_L)
    plt.ylabel(Text)
    plt.savefig(filename)
    plt.show()

    return
