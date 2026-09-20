'''
Модуль визначення оптимального ступіня полінома
Першоджерело: https://jrnl.nau.edu.ua/index.php/PIU/article/view/17001
'''

import numpy as np
import math as mt

def degree_polynomial_derivatives_scor(SL: np.ndarray, dt: float) -> int:

    '''
    Функція визначення оптимального ступіня полінома
    '''


    iter = len(SL)
    y1 = np.zeros((iter))
    y2 = np.zeros((iter))
    y3 = np.zeros((iter))
    y4 = np.zeros((iter))
    y5 = np.zeros((iter))

    # похідні високих порядків
    for i in range(iter):

        if i < (iter - 1):
            y1[i] = (SL[i+1] - SL[i]) / dt
        if i < (iter - 2):
            y2[i] = (SL[i+2] - 2 * SL[i+1] + SL[i]) / (dt ** 2)
        if i < (iter - 3):
            y3[i] = (SL[i+3] - 3 * SL[i+2] + 3 * SL[i+1] - SL[i]) / (dt ** 3)
        if i < (iter - 4):
            y4[i] = (SL[i+4] - 4 * SL[i+3] + 6 * SL[i+2] - 4 * SL[i+1] + SL[i]) / (dt ** 4)
        if i < (iter - 5):
            y5[i] = (SL[i+5] - 5 * SL[i+4] + 10 * SL[i+3] - 10 * SL[i+2] + 5 * SL[i+1] - SL[i]) / (dt ** 5)

    mean_array = np.zeros((5))
    mean_array[0] = abs(np.mean(y1))
    mean_array[1] = abs(np.mean(y2))
    mean_array[2] = abs(np.mean(y3))
    mean_array[3] = abs(np.mean(y4))
    mean_array[4] = abs(np.mean(y5))
    m_polinom_mean = np.argmin(mean_array) + 1
    print(' mean_array = ',  mean_array)
    print(' m_polinom_mean = ', m_polinom_mean)

    # mean_array = [0.00505324  0.0002327 - 0.00117747  0.00187607 - 0.00292356]

    # СКВ похідних експериментальне
    scv_y1_exp = mt.sqrt(np.var(y1))
    scv_y2_exp = mt.sqrt(np.var(y2))
    scv_y3_exp = mt.sqrt(np.var(y3))
    scv_y4_exp = mt.sqrt(np.var(y4))
    scv_y5_exp = mt.sqrt(np.var(y5))

    # СКВ похідних аналітичне
    var_SL = np.var(SL)
    scv_y1_theor = mt.sqrt((2 * var_SL) / (dt ** 2))
    scv_y2_theor = mt.sqrt((2 * var_SL) / (dt ** 4))
    scv_y3_theor = mt.sqrt((2 * var_SL) / (dt ** 6))
    scv_y4_theor = mt.sqrt((2 * var_SL) / (dt ** 8))
    scv_y5_theor = mt.sqrt((2 * var_SL) / (dt ** 10))

    # Розрахунок дельти
    dalta_array = np.zeros((5))
    dalta_array[0] = abs(scv_y1_exp - scv_y1_theor)
    dalta_array[1] = abs(scv_y2_exp - scv_y2_theor)
    dalta_array[2] = abs(scv_y3_exp - scv_y3_theor)
    dalta_array[3] = abs(scv_y4_exp - scv_y4_theor)
    dalta_array[4] = abs(scv_y5_exp - scv_y5_theor)

    # ступінь полінома, як мінімум на dalta_array
    m_polinom = np.argmin(dalta_array) + 1

    print('дельти = ',  dalta_array)
    print('Оптимальний ступінь полінома = ', m_polinom)


    return m_polinom
