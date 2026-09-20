'''
Приклади нормалізації
'''


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from data_parser import file_parsing



def data_plot(S_in, Text):
    '''
    Функція візуалізації
    '''

    plt.clf()
    plt.plot(S_in)
    plt.ylabel(Text)
    plt.show()


    return




def sum_normal(S_in):
    '''
    функція нормалізації відностносуми значень ряду
    '''

    n = len(S_in)
    S_sum_normal = np.zeros((n))
    sum_sum = np.sum(S_in)
    print('sum_sum =', sum_sum)
    for i in range(n):
        S_sum_normal[i] = S_in[i] / sum_sum


    return S_sum_normal



def max_normal(S_in):
    '''
    функція нормалізації відностно максимального значення
    '''

    n = len(S_in)
    S_max_normal = np.zeros((n))
    max_max = np.max(S_in)
    print('max_max =', max_max)
    for i in range(n):
        S_max_normal[i] = S_in[i] / max_max


    return S_max_normal




if __name__ == '__main__':

    path_data = 'data/'

    # вхідні дані
    S_in = file_parsing('https://www.oschadbank.ua/rates-archive', path_data + 'Oschadbank (USD).xls', 'Купівля')

    # препроцесінг / нормалізація - в "сирому" коді
    S_max_normal_out = max_normal(S_in)
    data_plot(S_in, 'Вхідні дані')
    data_plot(S_max_normal_out, 'Нормалізовані за max_normal дані')

    S_sum_normal_out = sum_normal(S_in)
    data_plot(S_sum_normal_out, 'Нормалізовані за sum_normal дані')


    # препроцесінг / нормалізація - в бібліотечному коді для глибинного навчання
    '''
    Докладна та корисна інструкція з препроцесінгу даних
    https://scikit-learn.org/stable/modules/preprocessing.html
    '''
    k = int(len(S_in)/2)
    X_train = S_in.reshape(k, 2)
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    print('X_train_minmax =', X_train_minmax)
    data_plot(X_train_minmax, 'Нормалізовані за MinMaxScaler')
    predicted = min_max_scaler.inverse_transform(X_train_minmax)
    data_plot(predicted, 'Відновлення з MinMaxScaler')


    '''
    На самостійне опрацювання - робота з numpy нормалізатором і не тільки
    https://www.geeksforgeeks.org/python/how-to-normalize-an-array-in-numpy-in-python/
    https://medium.com/@Doug-Creates/normalize-an-array-in-numpy-3bd5797fdfe9
    '''


