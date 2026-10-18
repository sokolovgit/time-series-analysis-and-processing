'''
Алгоритми ковзного середнього
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import math as mt
import warnings
warnings.filterwarnings('ignore')




def datanparray_parsing_and_plot(File_name, Data_name):
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



def data_plot_simslary(data_1, data_2, data_3, text_1, text_2, text_3):
    '''
    Відображення часових рядів
    '''

    plt.figure(figsize=(9, 3))
    plt.plot(data_1, label=text_1, color='r')
    plt.plot(data_2, label=text_2, color='g')
    plt.plot(data_3, label=text_3, color='b')

    plt.legend()
    plt.show()

    return



def data_plot_simslary_2d(data_1, data_2, text_1, text_2):
    '''
    Відображення часових рядів
    '''

    plt.figure(figsize=(9, 3))
    plt.plot(data_1, label=text_1, color='r')
    plt.plot(data_2, label=text_2, color='g')

    plt.legend()
    plt.show()

    return



def data_plot_3d(data_1, data_2, data_3, text_1, text_2, text_3):
    '''
    3d візуалізатор реальних даних
    '''

    xg = np.zeros((len(data_1)))
    for i in range(len(data_1)):
        xg[i] = i

    fig = pylab.figure()
    ax = fig.add_subplot(projection='3d')
    clr = ['g', 'r', 'b']
    ax.bar(xg, data_1, 1, zdir='y', color=clr[0])
    ax.bar(xg, data_2, 2, zdir='y', color=clr[1])
    ax.bar(xg, data_3, 3, zdir='y', color=clr[2])
    ax.set_xlabel('X Time')
    ax.set_ylabel('Y Data')
    ax.set_zlabel('Z quantity')

    pylab.show()

    return



def LSM(S0):

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

    return Yout[(iter-1), 0]
    # return Yout[0, 0]



def sliding_window(S0, n_Wind):
    '''
    Алгоритм ковзного середнього
    '''

    # параметри циклів
    iter = len(S0)
    j_Wind = mt.ceil(iter-n_Wind)+1
    S0_Wind = np.zeros((n_Wind))
    Midi = np.zeros((iter))

    #  ковзне вікно
    for j in range(j_Wind):
        for i in range(n_Wind):
            l = (j + i)
            S0_Wind[i] = S0[l]
        # Стат хар ковзного вікна
        Midi[l] = np.median(S0_Wind)

    # очищена вибірка
    S0_Midi = np.zeros((iter))
    for j in range(iter):
        S0_Midi[j] = Midi[j]

    for j in range(n_Wind):
        S0_Midi[j] = S0[j]

    return S0_Midi



def sliding_window_LSM(S0, n_Wind):
    '''
    Алгоритм ковзного середнього із МНК
    '''

    '''
       Алгоритм ковзного середнього
       '''

    # параметри циклів
    iter = len(S0)
    j_Wind = mt.ceil(iter - n_Wind) + 1
    S0_Wind = np.zeros((n_Wind))
    Midi = np.zeros((iter))

    #  ковзне вікно
    for j in range(j_Wind):
        for i in range(n_Wind):
            l = (j + i)
            S0_Wind[i] = S0[l]
        # Стат хар ковзного вікна
        Midi[l] = LSM(S0_Wind)

    # очищена вибірка
    S0_Midi = np.zeros((iter))
    for j in range(iter):
        S0_Midi[j] = Midi[j]

    for j in range(n_Wind):
        S0_Midi[j] = S0[j]

    return S0_Midi


if __name__ == '__main__':

    # datanparray_parsing
    data_np_1 = datanparray_parsing_and_plot('data/Oschadbank (USD).xls', 'Купівля')
    data_np_2 = datanparray_parsing_and_plot('data/Oschadbank (USD).xls', 'Продаж')
    data_np_3 = datanparray_parsing_and_plot('data/Oschadbank (USD).xls', 'КурсНбу')
    data_plot_3d(data_np_1, data_np_2, data_np_3, 'Купівля', 'Продаж', 'КурсНбу')
    data_plot_simslary(data_np_1, data_np_2, data_np_3, 'Купівля', 'Продаж', 'КурсНбу')

    '''
    Купівля
    '''
    # ковзне середнє
    data_sw = sliding_window(data_np_1, 5)
    data_plot_simslary_2d(data_np_1, data_sw, 'Купівля', 'sliding_window')
    # ковзне середнє із МНК
    data_sw_lsm = sliding_window_LSM(data_np_1, 20)
    data_plot_simslary_2d(data_np_1, data_sw_lsm, 'Купівля', 'sliding_window_lsm')
    # ВІЗУАЛІЗАТОР: ковзне середнє + ковзне середнє із МНК
    data_plot_simslary(data_np_1, data_sw, data_sw_lsm, 'Купівля', 'sliding_window', 'sliding_window_lsm')

    '''
    Продаж
    '''
    # ковзне середнє
    data_sw = sliding_window(data_np_2, 5)
    data_plot_simslary_2d(data_np_2, data_sw, 'Купівля', 'sliding_window')
    # ковзне середнє із МНК
    data_sw_lsm = sliding_window_LSM(data_np_2, 20)
    data_plot_simslary_2d(data_np_2, data_sw_lsm, 'Купівля', 'sliding_window_lsm')
    # ВІЗУАЛІЗАТОР: ковзне середнє + ковзне середнє із МНК
    data_plot_simslary(data_np_2, data_sw, data_sw_lsm, 'Купівля', 'sliding_window', 'sliding_window_lsm')


    '''
    КурсНбу
    '''
    # ковзне середнє
    data_sw = sliding_window(data_np_3, 5)
    data_plot_simslary_2d(data_np_3, data_sw, 'Купівля', 'sliding_window')
    # ковзне середнє із МНК
    data_sw_lsm = sliding_window_LSM(data_np_3, 20)
    data_plot_simslary_2d(data_np_3, data_sw_lsm, 'Купівля', 'sliding_window_lsm')
    # ВІЗУАЛІЗАТОР: ковзне середнє + ковзне середнє із МНК
    data_plot_simslary(data_np_3, data_sw, data_sw_lsm, 'Купівля', 'sliding_window', 'sliding_window_lsm')








