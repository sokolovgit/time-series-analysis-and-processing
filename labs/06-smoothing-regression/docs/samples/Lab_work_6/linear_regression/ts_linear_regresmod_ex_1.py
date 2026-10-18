'''
Лінійна регресія з МНК
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')



def dataframe_parser_and_plot(path_data, filename, ind_x, k):
    '''
    Парсінг вхідних даних з файлу в pandas dataframe

    '''

    dateparse = lambda x: pd.to_datetime(x, dayfirst=True)
    d = pd.read_excel((path_data + filename), index_col=ind_x, date_parser=dateparse)
    ts = d.iloc[:, k]

    print('d = ', d)
    print('ts= ', ts)

    plt.plot(ts, label='Original Time Series')
    plt.legend()
    plt.show()

    return ts


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


def restoration_of_measurement_gaps(data_in_real):
    '''
    Наївне відновлення пропусків шляхлм заміни попередніми даними
    '''

    data_in_rezalt = np.zeros((len(data_in_real)))

    for i in range(len(data_in_real)):
        if data_in_real[i] == 0:
            data_in_rezalt[i] = data_in_rezalt[i - 1]
        else:
            data_in_rezalt[i] = data_in_real[i]

    return data_in_rezalt



def LSM(S0):
    '''
    Лінійна регресія за МНК
    '''

    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 6))

    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
        F[i, 3] = float(i * i * i)
        F[i, 4] = float(i * i * i * i)
        F[i, 5] = float(i * i * i * i * i)

    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)

    print('Регресійна модель:')
    print('y(t) = ', C[0, 0], ' + ', C[1, 0], ' * t', ' + ', C[2, 0], ' * t^2', ' + ', C[3, 0], ' * t^3',
          ' + ', C[4, 0], ' * t^4', ' + ', C[5, 0], ' * t^5')

    return Yout



def metrics_predict(real_data, predicted, text):

    '''
    Оцінювання метрік результатів прогнозування з нейромережею
    '''

    # Обчислення метрик якості моделі на прогнозуванні
    mae = mean_absolute_error(real_data, predicted)
    mse = mean_squared_error(real_data, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(real_data, predicted)

    print('\n')
    print(text)
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R2) Score: {r2}")

    return




if __name__ == '__main__':

    # dataframe_parser: 1 - Купівля, 2 - Продаж, 3 - КурсНбу
    ts_1 = dataframe_parser_and_plot('data/', 'Oschadbank (USD).xls', 'Дата', 1)
    ts_2 = dataframe_parser_and_plot('data/', 'Oschadbank (USD).xls', 'Дата', 2)
    ts_3 = dataframe_parser_and_plot('data/', 'Oschadbank (USD).xls', 'Дата', 3)

    # datanparray_parsing
    data_np_1 = datanparray_parsing_and_plot('data/Oschadbank (USD).xls', 'Купівля')
    data_np_2 = datanparray_parsing_and_plot('data/Oschadbank (USD).xls', 'Продаж')
    data_np_3 = datanparray_parsing_and_plot('data/Oschadbank (USD).xls', 'КурсНбу')

    # візуалізатор
    data_plot_3d(data_np_1, data_np_2, data_np_3, 'Купівля', 'Продаж', 'КурсНбу')
    data_plot_simslary(data_np_1, data_np_2, data_np_3, 'Купівля', 'Продаж', 'КурсНбу')


    # відновлення  пропусків
    data_np_1_restoration = restoration_of_measurement_gaps(data_np_1)
    data_np_2_restoration = restoration_of_measurement_gaps(data_np_2)
    data_np_3_restoration = restoration_of_measurement_gaps(data_np_3)

    # візуалізатор
    data_plot_3d(data_np_1_restoration, data_np_2_restoration, data_np_3_restoration, 'Купівля', 'Продаж', 'КурсНбу')
    data_plot_simslary(data_np_1_restoration, data_np_2_restoration, data_np_3_restoration, 'Купівля', 'Продаж', 'КурсНбу')

    # лінійна регресія за МНК
    data_np_1_restoration_LSM = LSM(data_np_1_restoration)
    metrics_predict(data_np_1_restoration, data_np_1_restoration_LSM, 'Купівля')
    data_np_2_restoration_LSM = LSM(data_np_2_restoration)
    metrics_predict(data_np_2_restoration, data_np_2_restoration_LSM, 'Продаж')
    data_np_3_restoration_LSM = LSM(data_np_3_restoration)
    metrics_predict(data_np_3_restoration, data_np_3_restoration_LSM, 'КурсНбу')
    # візуалізатор
    data_plot_simslary(data_np_1_restoration_LSM, data_np_2_restoration_LSM,
                       data_np_3_restoration_LSM, 'Купівля', 'Продаж', 'КурсНбу')






