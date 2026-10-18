'''
Аналіз динамічних властивостей часового ряду з фрактальними показниками
https://github.com/Mottl/hurst
http://dwl.kiev.ua/art/oiad/oiad.pdf

'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hurst import compute_Hc



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



def  hurst_index(data_np):
    '''
    Аналіз показника Херста

    '''

    series = pd.Series(data_np)

    H, c, data = compute_Hc(series, kind='price', simplified=True)

    print("H={:.4f}, c={:.4f}".format(H, c))

    f, ax = plt.subplots()
    ax.plot(data[0], c * data[0] ** H, color="deepskyblue")
    ax.scatter(data[0], data[1], color="purple")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Time interval')
    ax.set_ylabel('R/S ratio')
    ax.grid(True)
    plt.show()

    return




if __name__ == '__main__':

    data_np_1 = file_parsing('data/Oschadbank (USD).xls', 'Купівля')
    # data_np_2 = file_parsing('data/Oschadbank (USD).xls', 'Продаж')
    # data_np_3 = file_parsing('data/Oschadbank (USD).xls', 'КурсНбу')

    hurst_index(data_np_1)


    '''
    Значення H > 1/ 2 означає, що спрямована в певну
    сторону динаміка процесу в минулому, найімовірніше, спричинить
    продовження руху у тому ж напрямку. 
    Якщо H  < 1/ 2, то прогнозується, що процес змінить спрямованість. 
    H  = 1/ 2 означає невизначеність –
    броунівський рух.

    '''

