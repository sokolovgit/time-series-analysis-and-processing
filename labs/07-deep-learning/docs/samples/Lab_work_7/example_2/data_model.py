'''
Модуль генерації синтезованих / модельних даних
'''

import numpy as np
import math as mt
import matplotlib.pyplot as plt



def randomAM(n, iter, nAV):
    '''
    рівномірний закон розводілу номерів АВ в межах вибірки
    '''

    SAV = np.zeros((nAV))
    S = np.zeros((n))

    for i in range(n):
        S[i] = np.random.randint(0, iter)  # параметри закону задаются межами аргументу
    mS = np.median(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)

    # генерація номерів АВ за рівномірним законом
    for i in range(nAV):
        SAV[i] = mt.ceil(np.random.randint(1, iter))  # рівномірний розкид номерів АВ в межах вибірки розміром 0-iter
    print('номери АВ: SAV=', SAV)
    print('----- статистичны характеристики РІВНОМІРНОГО закону розподілу ВВ -----')
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('-----------------------------------------------------------------------')

    # гістограма закону розподілу ВВ
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.show()

    return SAV



def randoNORM(dm, dsig, iter):
    '''
    нормальний закон розводілу ВВ
    '''

    S = np.random.normal(dm, dsig, iter)  # нормальний закон розподілу ВВ з вибіркою єбємом iter та параметрами: dm, dsig
    mS = np.median(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)

    print('------- статистичны характеристики НОРМАЛЬНОЇ похибки вимірів -----')
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('------------------------------------------------------------------')

    # гістограма закону розподілу ВВ
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.show()

    return S



def Model_real(n):
    '''
    модель ідеального тренду синтезованих даних:
    отримана за МНК для конкретних даних і потребує корекції при їх зміні
    '''

    S0=np.zeros((n))

    for i in range(n):
        S0[i] = 28.03423083418919 + (0.004773157735137576 * i) + (0.0008240042970011226 * (i ** 2)) - (
                    3.204450289838111e-06 * (i ** 3)) + (7.543074486030252e-10 * (i ** 4)) + (
                            6.87528764332385e-12 * (i ** 5))

    return S0



def Model_NORM(SN, S0N, n):
    '''
    модель виміру (тренду) з нормальний шумом
    '''

    SV=np.zeros((n))

    for i in range(n):
        SV[i] = S0N[i]+SN[i]

    return SV



def Model_NORM_AV(S0, SV, nAV, Q_AV, dm,  dsig, SAV):
    '''
    модель виміру (тренду) з нормальний шумом + АНОМАЛЬНІ ВИМІРИ
    '''

    SV_AV = SV
    SSAV = np.random.normal(dm, (Q_AV * dsig), nAV)  # аномальна випадкова похибка з нормальним законом

    for i in range(nAV):
        k = int(SAV[i])
        SV_AV[k] = S0[k] + SSAV[i]        # аномальні вимірів з рівномірно розподіленими номерами

    return SV_AV
