'''
Дослідження часових рядів методами візуалізації: 2d графік, гістограмма
https://www.oreilly.com/library/view/practical-time-series/9781492041641/

'''


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')



def data_parser_and_plot(path_data, filename, ind_x, k):
    '''
    Парсінг вхідних даних з файлу в pandas dataframe - побудова графіка

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

    plt.scatter(x, S_real)
    plt.title("Basic Scatter Plot")
    plt.xlabel("X Values")
    plt.ylabel("Y Values")
    plt.show()

    plt.figure(figsize=(15, 5))
    plt.plot(x, S_real, 'o')  # 'o' specifies circular markers
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Plotting Individual Points")
    plt.grid(True)
    plt.show()


    plt.clf()
    plt.plot(S_real)
    plt.show()

    plt.hist(S_real, bins=50, facecolor="blue", alpha=0.5)
    plt.show()


    return S_real



if __name__ == '__main__':


    ts = data_parser_and_plot('data/', 'Oschadbank (USD).xls', 'Дата', 1)  # 1 - Купівля, 2 - Продаж, 3 - КурсНбу
    ts = data_parser_and_plot('data/', 'usd_2023-2024_1.xlsx', 'Дата', 5)

    data_np = file_parsing('data/Oschadbank (USD).xls', 'Купівля')



