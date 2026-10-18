'''
Авторегресія

'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
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



def ABF(S0):
    '''
    алгоритм -а-b фільтрa
    '''

    iter = len(S0)
    Yin = np.zeros((iter, 1))
    YoutAB = np.zeros((iter, 1))
    T0 = 1

    for i in range(iter):
        Yin[i, 0] = float(S0[i])

    # початкові дані для запуску фільтра
    Yspeed_retro = (Yin[1, 0] - Yin[0, 0]) / T0
    Yextra = Yin[0, 0] + Yspeed_retro
    alfa = 2 * (2 * 1 - 1) / (1 * (1 + 1))
    beta = (6 / 1) * (1 + 1)
    YoutAB[0, 0] = Yin[0, 0] + alfa * (Yin[0, 0])

    # рекурентний прохід по вимірам
    for i in range(1, iter):
        YoutAB[i, 0] = Yextra + alfa * (Yin[i, 0] - Yextra)
        Yspeed = Yspeed_retro + (beta / T0) * (Yin[i, 0] - Yextra)
        Yspeed_retro = Yspeed
        Yextra = YoutAB[i, 0] + Yspeed_retro
        alfa = (2 * (2 * i - 1)) / (i * (i + 1))
        beta = 6 / (i * (i + 1))

        # print('Регресійна модель ab-filtr:')
        # print('y(t) = ',alfa, ' + ',  beta, ' * t')

    return YoutAB



def ABGF(S0):
    '''
    алгоритм -а-b-g фільтрa
    '''

    iter = len(S0)
    Yin = np.zeros((iter, 1))
    Yout_ABG = np.zeros((iter, 1))
    T0 = 1
    for i in range(iter):
        Yin[i, 0] = float(S0[i])

    # початкові дані для запуску фільтра
    Yspeed_retro = (Yin[1, 0] - Yin[0, 0]) / T0
    Yaccel_retro = (Yin[2, 0] - 2 * Yin[1, 0] + Yin[0, 0]) / (T0 * T0)
    Yextra = Yin[0, 0] + (Yspeed_retro * T0) + (Yaccel_retro * 0.5 * T0 * T0)

    # alpha, beta, gamma
    alpha = 3 * (3 * 1 * 1 - 3 * 1 + 2) / (1 * (1 + 1) * (1 + 2))
    beta = 18 * (2 * 1 - 1) / (T0 * (1 + 1) * (1 + 2) * 1)
    gamma = 60 / (T0 * T0 * (1 + 1) * (1 + 2) * 1)
    Yout_ABG[0, 0] = Yin[0, 0]

    #  рекурентний прохід по вимірам
    for i in range(1, iter):
        Yout_ABG[i, 0] = Yextra + alpha * (Yin[i, 0] - Yextra)
        Yspeed = Yspeed_retro + (beta / T0) * (Yin[i, 0] - Yextra)
        Yaccel = Yaccel_retro + (gamma / T0 * T0) * (Yin[i, 0] - Yextra)
        Yspeed_retro = Yspeed
        Yextra = Yout_ABG[i, 0] + (Yspeed_retro * T0) + (Yaccel * 0.5 * T0 * T0)

        # alpha, beta, gamma
        alpha = 3 * (3 * i * i - 3 * i + 2) / (i * (i + 1) * (i + 2))
        beta = 18 * (2 * i - 1) / (T0 * (i + 1) * (i + 2) * i)
        gamma = 60 / (T0 * T0 * (i + 1) * (i + 2) * i)

    return Yout_ABG


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
    data_ABF = ABF(data_np_1)
    data_plot_simslary_2d(data_np_1, data_ABF, 'Купівля', 'sdata_ABF')
    data_ABGF = ABGF(data_np_1)
    data_plot_simslary_2d(data_np_1, data_ABGF, 'Купівля', 'data_ABGF')
    data_plot_simslary(data_np_1, data_ABF, data_ABGF, 'Купівля', 'sdata_ABF', 'data_ABGF')

    '''
    Продаж
    '''
    data_ABF = ABF(data_np_2)
    data_plot_simslary_2d(data_np_2, data_ABF, 'Купівля', 'sdata_ABF')
    data_ABGF = ABGF(data_np_2)
    data_plot_simslary_2d(data_np_2, data_ABGF, 'Купівля', 'data_ABGF')
    data_plot_simslary(data_np_2, data_ABF, data_ABGF, 'Купівля', 'sdata_ABF', 'data_ABGF')


    '''
    КурсНбу
    '''
    data_ABF = ABF(data_np_3)
    data_plot_simslary_2d(data_np_3, data_ABF, 'Купівля', 'sdata_ABF')
    data_ABGF = ABGF(data_np_3)
    data_plot_simslary_2d(data_np_3, data_ABGF, 'Купівля', 'data_ABGF')
    data_plot_simslary(data_np_3, data_ABF, data_ABGF, 'Купівля', 'sdata_ABF', 'data_ABGF')

