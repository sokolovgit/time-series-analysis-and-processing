'''
Множинна регресія:
'''


import pandas as pd
import matplotlib.pyplot as plt
import pandas
from sklearn import linear_model
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


def multiple_linear_regression(filename):
    '''
    Тестовий приклад множинної регресії:
    https://www.w3schools.com/python/python_ml_multiple_regression.asp
    Вхідні дані:
    data/data.csv
    Ми можемо передбачити викиди CO2 автомобілем на основі розміру двигуна,
    але за допомогою множинної регресії ми можемо додати більше змінних, таких як вага автомобіля,
    щоб зробити прогноз точнішим.

    '''


    df = pandas.read_csv(filename)

    X = df[['Weight', 'Volume']]
    y = df['CO2']

    regr = linear_model.LinearRegression()
    regr.fit(X, y)

    # predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
    predictedCO2 = regr.predict([[2300, 1300]])

    print('predictedCO2 = ', predictedCO2)

    return



def multiple_linear_regression_oshaddbank(filename, purchase, sale):
    '''
    Приклад множинної регресії для задачі:
    Передбачити курс Нбу за параметрами Купівля, Продаж

    '''

    df = pd.read_excel(filename)

    X = df[['Купівля', 'Продаж']]
    y = df['КурсНбу']


    regr = linear_model.LinearRegression()
    regr.fit(X, y)

    # прогнозування
    predicted_oshaddbank = regr.predict([[purchase, sale]])

    print('predicted_oshaddbank = ', predicted_oshaddbank)


    return

if __name__ == '__main__':


    # dataframe_parser: 1 - Купівля, 2 - Продаж, 3 - КурсНбу
    ts_1 = dataframe_parser_and_plot('data/', 'Oschadbank (USD).xls', 'Дата', 1)
    ts_2 = dataframe_parser_and_plot('data/', 'Oschadbank (USD).xls', 'Дата', 2)
    ts_3 = dataframe_parser_and_plot('data/', 'Oschadbank (USD).xls', 'Дата', 3)

    data_plot_simslary(ts_1, ts_2, ts_3, 'Купівля', 'Продаж', 'КурсНбу')

    multiple_linear_regression_oshaddbank('data/Oschadbank (USD).xls', 30, 30)

    multiple_linear_regression('data/data.csv')


