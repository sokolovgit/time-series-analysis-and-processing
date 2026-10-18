'''
Модель згорткової нейронної мережі для обробки та прогнозування часових рядів.
Автори: Олексій Писарчук, Данило Баран, права авторства - захищено.
Не півдлягає поширенню без посилання на авторів.

Функціонал:
1. Згорткова нейронна мережа, структура опрацьована методами тестування;
2. Вхідні дані - поділ на порції забезпечує високі показники якості моделі;
3. Вхідний прошарок побудовано за оптимізаційною моделлю неліліної схеми компромісів;
4. Прогнозування поза межіми інтервалу спостереження реалізовано на синтезованій моделі з трендом від МНК.

'''


import numpy as np
from data_parser import file_parsing
from data_model import Model_real, randomAM, randoNORM, Model_NORM, Model_NORM_AV
from data_analytics import Plot_AV,  Stat_characteristics_in
from ts_neural_networks import convolutional_neural_network_model




if __name__ == '__main__':


    # ініциалізація сховищ результатів
    path_plot = 'data_plot/'
    path_data = 'data/'
    filename_hist = path_plot + 'data_hist_in'


    # реальні вхіднi данi
    SV_AV_1 = file_parsing('https://www.oschadbank.ua/rates-archive', path_data + 'Oschadbank (USD).xls',
                           'Купівля')  # реальні дані
    # аналіз вхідних даних
    n = len(SV_AV_1)
    iter_in = int(len(SV_AV_1))  # кількість реалізацій ВВ
    Plot_AV(SV_AV_1, SV_AV_1, 'Коливання курсу USD в 2022 році за даними Ощадбанк', path_plot + 'data_real.png')
    Stat_characteristics_in(SV_AV_1, 'Коливання курсу USD в 2022 році за даними Ощадбанк', filename_hist)


    # API констант для синтезованих даних
    Q_AV = 3  # коефіцієнт переваги АВ
    nAVv = 10
    nAV = int((iter_in * nAVv) / 100)  # кількість АВ у відсотках та абсолютних одиницях
    dm = 0
    dsig = 0.99457  # параметри нормального закону розподілу ВВ: середне та СКВ
    dsig = 0.1  # параметри нормального закону розподілу ВВ: середне та СКВ


    # API констант для нейромережі
    k_expo = 100  # кількість точок екстраполяції
    sigma = 1
    batch_size = 200  # поділ вхідного датасет на порції
    epochs = 100  # епохи
    n_neurons = [512, 256, 128, 64]  # параметри архітектури мережі
    '''
    Встановлення параметрів нейромережі за бінарним рядом
    2**9 = 512
    2**8 = 256
    2**7 = 128
    2**6 =64
    '''

    # модель синтезованих даних
    S0 = Model_real(n)  # модель ідеального тренду (квадратичний закон)
    SAV = randomAM(n, iter_in, nAV)  # модель рівномірних номерів АВ
    S = randoNORM(dm, dsig, iter_in)  # модель нормальних помилок
    # Нормальні похибки
    SV = Model_NORM(S, S0, n)  # модель тренда + нормальних помилок
    Plot_AV(S0, SV, 'квадратична модель + Норм. шум', path_plot + 'Model_NORM.png')
    Stat_characteristics_in(SV, 'Вибірка + Норм. шум', filename_hist)
    # Аномальні похибки
    SV_AV = Model_NORM_AV(S0, SV, nAV, Q_AV, dm, dsig, SAV)  # модель тренда + нормальних помилок + АВ
    # аналіз синтезованих даних
    Plot_AV(S0, SV_AV, 'квадратична модель + Норм. шум + АВ', path_plot + 'Model_NORM_AV.png')
    Stat_characteristics_in(SV_AV, 'Вибірка з АВ', filename_hist)

    print('УВАГА дані')
    print('Реальні дані: ', SV_AV_1)
    print('Синтезованіі дані: ', SV_AV)


    # змішування для нейромережі реальних та синтезованих даних
    S_in = np.zeros((n))
    for i in range(n):
        S_in[i] = SV_AV_1[i]
        if i > (n - k_expo):
            S_in[i] = SV_AV[i]

    print(' Реальні + Синтезованіі дані: ', S_in)
    Plot_AV(S_in, S_in, 'Реальні + Синтезованіі дані', path_plot + 'Model_and_real_data.png')


    # Головні виклики для нейромережі
    real_data = SV_AV_1
    data_predicted = SV_AV_1    # для прогнозування використані реальні дані
    data_predicted = S_in       # для прогнозування використані МНК дані
    convolutional_neural_network_model(batch_size, epochs, real_data, data_predicted, n_neurons, k_expo, sigma)
