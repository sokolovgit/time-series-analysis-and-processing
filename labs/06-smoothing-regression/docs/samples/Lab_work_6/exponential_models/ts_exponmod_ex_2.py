'''
Візуальний аналіз
Подібність часових рядів / Similarity of time series
https://www.geeksforgeeks.org/machine-learning/similarity-search-for-time-series-data/
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.api import Holt
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tools.eval_measures import rmse
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


def Holt_Winter_Seasonal_Smoothing(data):
    '''
    Експоненціальне згладжування Холта-Вінтерса - Сезонна модель Холта-Вінтерса

    '''

    model = ExponentialSmoothing(
        data,
        seasonal_periods=24,
        trend='add',
        seasonal='add')

    model_single_fit = model.fit()


    forecast_single = model_single_fit.forecast(len(data))

    plt.plot(data, label='Original Data')
    plt.plot(model_single_fit.fittedvalues, label='Fitted Values')
    plt.plot(forecast_single, label='Forecast')
    plt.xlabel('Year')
    plt.ylabel('Number of Passengers')
    plt.title('Single Exponential Smoothing')
    plt.legend()
    plt.show()

    print('Експоненціальне згладжування Холта-Вінтерса - Сезонна модель Холта-Вінтерса')
    # Calculate mean squared error
    mean_squared_error_out = mean_squared_error(data, model_single_fit.fittedvalues)
    # sklearn.metrics.r2_score
    r2_score_out = r2_score(data, model_single_fit.fittedvalues)
    print('mean_squared_error_out = ', mean_squared_error_out)
    print('r2_score_out = ', r2_score_out, '\n')

    return




if __name__ == '__main__':

    # dataframe_parser: 1 - Купівля, 2 - Продаж, 3 - КурсНбу
    ts_1 = dataframe_parser_and_plot('data/', 'Oschadbank (USD).xls', 'Дата', 1)
    ts_2 = dataframe_parser_and_plot('data/', 'Oschadbank (USD).xls', 'Дата', 2)
    ts_3 = dataframe_parser_and_plot('data/', 'Oschadbank (USD).xls', 'Дата', 1)

    # datanparray_parsing
    data_np_1 = datanparray_parsing_and_plot('data/Oschadbank (USD).xls', 'Купівля')
    data_np_2 = datanparray_parsing_and_plot('data/Oschadbank (USD).xls', 'Продаж')
    data_np_3 = datanparray_parsing_and_plot('data/Oschadbank (USD).xls', 'КурсНбу')

    data_plot_3d(data_np_1, data_np_2, data_np_3, 'Купівля', 'Продаж', 'КурсНбу')

    data_plot_simslary(data_np_1, data_np_2, data_np_3, 'Купівля', 'Продаж', 'КурсНбу')


    # Експоненціальне згладжування Холта-Вінтерса - Сезонна модель Холта-Вінтерса
    Holt_Winter_Seasonal_Smoothing(ts_1)

