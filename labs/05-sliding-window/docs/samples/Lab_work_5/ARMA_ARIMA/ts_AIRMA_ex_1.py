'''
Алгоритм ARІMA - для Ощадбанка
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
from statsmodels.tsa.arima.model import ARIMA
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

    plt.figure(figsize=(12, 3))
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



def main_ARIMA(data):


    time_series = data

    # Розділення даних на навчальні та тестові набори (необов'язково, але гарна практика)
    train_size = int(len(time_series) * 0.7)
    train_data, test_data = time_series[:train_size], time_series[train_size:]

    # Встановлення параметрів
    order = (1, 1, 15)

    # 3.Формування моделі ARIMA
    model = ARIMA(train_data, order=order)
    # model = ARIMA(train_data, order=(0, 1, 0), seasonal_order=(2, 1, 1, 12))

    model_fit = model.fit()

    # КРІ моделі
    print(model_fit.summary())

    # 4.Прогнозування

    forecast_steps = len(test_data)
    forecast = model_fit.forecast(steps=forecast_steps)


    forecast1 = model_fit.forecast(steps=200)
    data_plot_simslary(time_series, time_series, forecast1, 'train_data', 'train_data', 'forecast1')

    # 5. Візуалізатор
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data, label='Training Data', color='blue')
    plt.plot(test_data.index, test_data, label='Actual Test Data', color='green')
    plt.plot(forecast.index, forecast, label='ARIMA Forecast', color='red', linestyle='--')
    plt.title('ARIMA Time Series Forecast')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()


    return  forecast


if __name__ == '__main__':

    # dataframe_parser: 1 - Купівля, 2 - Продаж, 3 - КурсНбу
    ts_1 = dataframe_parser_and_plot('data/', 'Oschadbank (USD).xls', 'Дата', 1)
    ts_2 = dataframe_parser_and_plot('data/', 'Oschadbank (USD).xls', 'Дата', 2)
    ts_3 = dataframe_parser_and_plot('data/', 'Oschadbank (USD).xls', 'Дата', 3)



    main_ARIMA(ts_1)
    main_ARIMA(ts_2)
    main_ARIMA(ts_3)

