'''
Декомпозиція часових рядів / Time Series Decomposition
https://www.geeksforgeeks.org/python/time-series-decomposition-techniques/
'''


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')



def data_parser_and_plot(path_data, filename, ind_x, k):
    '''
    Парсінг вхідних даних з файлу в pandas dataframe

    '''

    dateparse = lambda x: pd.to_datetime(x, dayfirst=True)
    d = pd.read_excel((path_data + filename), index_col=ind_x, date_parser=dateparse)
    ts = d.iloc[:, k]  # 1 - Купівля, 2 - Продаж, 3 - КурсНбу

    print('d = ', d)
    print('ts= ', ts)

    plt.figure(figsize=(12, 3))
    plt.plot(ts, label='Original Time Series')
    plt.legend()
    plt.show()

    return ts



def ts_decomposition(ts, ts_model, ts_period, text):
    '''
    Розкладання часового ряду в аддитивну / мультипликативну модель
    параметр period=4 (наприклад 4)- задає астрономічну сезоність, якщо дата не pandas формату: 2022-02-08

    '''

    result = seasonal_decompose(ts, model=ts_model, period=ts_period)
    plt.figure(figsize=(9, 3))
    plt.plot(result.trend, label=text)
    plt.legend()
    plt.show()

    return result




def ts_decomposition_seasonal(result, text):
    '''
    Аналіз сезонності
    '''

    ts_seasonal = result.seasonal
    plt.figure(figsize=(9, 3))
    plt.plot(ts_seasonal, label='Additive Seasonal')
    plt.legend()
    plt.show()

    return ts_seasonal



def data_plot_decomposition(result_add, result_mult):
    '''
    Відображення аддитивної та мультипликативної моделей декомпозиції
    '''

    plt.figure(figsize=(9, 3))
    plt.plot(ts, label='Original Time Series')
    plt.plot(result_add.trend, label='Additive Trend', color='g')
    plt.plot(result_mult.trend, label='Multiplicative Trend', color='r')
    plt.legend()
    plt.show()

    return



def data_plot_seasonal(result_add_seasonal, result_mult_seasonal):
    '''
    Відображення аддитивної та мультипликативної моделей декомпозиції
    '''

    plt.figure(figsize=(9, 3))
    plt.plot(result_add_seasonal, label='Additive Seasonal', color='g')
    plt.plot(result_mult_seasonal, label='Multiplicative Seasonal', color='r')
    plt.legend()
    plt.show()

    return


def ts_residual(ts):
    '''
    Розрахунок / візуалізатор залишків з ковзним вікном 2-х типів
    '''

    #  Розрахунок простої ковзної середньої (SMA) - ковзне вікно дає множину / ряд значень
    sma_window = 7  # 7-day moving average
    sma = ts.rolling(window=sma_window).mean()
    print('sma : ', sma)

    # Розрахунок експоненціальної ковзної середньої (EMA) - ковзне вікно дає множину / ряд значень
    ema_window = 30  # 30-day moving average
    ema = ts.ewm(span=ema_window, adjust=False).mean()
    print('ema : ', ema)

    #
    plt.figure(figsize=(9, 3))
    plt.plot(ts, label='Original Time Series')
    plt.plot(sma, label=f'{sma_window}-Day SMA')
    plt.plot(ema, label=f'{ema_window}-Day EMA')
    plt.legend()
    plt.show()

    return sma, ema




if __name__ == '__main__':

    # 1. Парсінг вхідних даних з файлу в pandas dataframe
    ts = data_parser_and_plot('data/', 'Oschadbank (USD).xls', 'Дата', 1)  # 1 - Купівля, 2 - Продаж, 3 - КурсНбу
    ts = data_parser_and_plot('data/', 'usd_2023-2024_1.xlsx', 'Дата', 5)

    # 2. Розкладання часового ряду в аддитивну / мультипликативну модель
    result_add = ts_decomposition(ts, 'additive', 4, 'Additive Trend')
    result_mult = ts_decomposition(ts, 'multiplicative', 4, 'Multiplicative Trend')
    # Відображення аддитивної та мультипликативної моделей декомпозиції
    data_plot_decomposition(result_add, result_mult)

    # 3. Аналіз сезонності
    ts_seasonal_add = ts_decomposition_seasonal(result_add, 'Additive Seasonal')
    ts_seasonal_mult = ts_decomposition_seasonal(result_mult, 'Multiplicative Seasonal')
    # Відображення аддитивної та мультипликативної сезонності
    data_plot_seasonal(ts_seasonal_add, ts_seasonal_mult)

    # 4. Аналіз залишків - шуму
    sma, ema = ts_residual(ts)
