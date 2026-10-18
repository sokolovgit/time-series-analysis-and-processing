'''
Властивості часових рядів / Time Series Properties

'''


import pandas as pd
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



def file_parsing(File_name, Data_name):
    '''
    Парсінг вхідних даних з файлу в numpy array - побудова гістограми

    '''

    d = pd.read_excel(File_name)
    for name, values in d[[Data_name]].items():
        print(values)
    S_real = np.zeros((len(values)))
    x = np.zeros((len(values)))

    for i in range(len(values)):
        S_real[i] = values[i]
        x[i] = i


    plt.clf()
    plt.plot(S_real)
    plt.show()


    return S_real



# статистичні характеристики вхідної вибірки
def ts_stat_characteristics(SL, text):

    # статистичні характеристики вибірки з урахуванням тренду
    Yout = LSM_Stat_characteristics(SL)
    iter = len(Yout)
    SL0 = np.zeros((iter))

    for i in range(iter):
        SL0[i] = SL[i] - Yout[i, 0]

    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)

    print(text)
    print('кількість елементів вбірки=', iter)
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS, '\n')


    plt.hist(SL0, bins=50, facecolor="blue", alpha=0.5)
    plt.show()

    return


def LSM_Stat_characteristics (S0):

    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 5))

    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        F[i, 1] = float(i)
        F[i, 2] = float(i ** 2)
        F[i, 3] = float(i ** 3)
        F[i, 4] = float(i ** 4)

    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)

    return Yout

if __name__ == '__main__':

    data_np_1 = file_parsing('data/Oschadbank (USD).xls', 'Купівля')
    data_np_2 = file_parsing('data/Oschadbank (USD).xls', 'Продаж')
    data_np_3 = file_parsing('data/Oschadbank (USD).xls', 'КурсНбу')

    ts_stat_characteristics(data_np_1, 'Купівля')
    ts_stat_characteristics(data_np_2, 'Продаж')
    ts_stat_characteristics(data_np_3, 'КурсНбу')


