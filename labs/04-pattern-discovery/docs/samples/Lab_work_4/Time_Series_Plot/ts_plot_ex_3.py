'''
Дослідження часових рядів методами візуалізації: теплова карта / heat map
https://www.oreilly.com/library/view/practical-time-series/9781492041641/
https://www.geeksforgeeks.org/python/how-to-draw-2d-heatmap-using-matplotlib-in-python/

'''



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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


    return d



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



if __name__ == '__main__':


    '''
    Кластерна хітмапа
    '''

    # абстрактний приклад
    df = sns.load_dataset("tips")
    heatmap_data = df.pivot_table(index="day", columns="sex", values="tip", aggfunc="mean")
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu")
    plt.show()

    # реальні дані - логіка сумнівна
    df = data_parser_and_plot('data/', 'Oschadbank (USD).xls', 'Дата', 1)  # 1 - Купівля, 2 - Продаж, 3 - КурсНбу
    df = data_parser_and_plot('data/', 'Oschadbank (USD).xls', 'Дата', 2)
    heatmap_data = df.pivot_table(index="Дата", columns="Купівля", values="Продаж", aggfunc="mean")
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu")
    plt.show()



    '''
    Докладна хітмапа
    '''

    # абстрактний приклад
    data = np.random.random((12, 12))
    plt.imshow(data, cmap='autumn')
    plt.title("Heatmap with different color")
    plt.show()


    # реальні дані - логіка сумнівна
    data_np_x = file_parsing('data/Oschadbank (USD).xls', 'Купівля')
    data_np_y = file_parsing('data/Oschadbank (USD).xls', 'Продаж')
    data = np.zeros((len(data_np_x), len(data_np_x)))
    for i in range(len(data_np_x)):
        for j in range(len(data_np_x)):
            data[i, j] = data_np_x[i]
    plt.imshow(data, cmap='autumn')
    plt.title("Heatmap with different color")
    plt.show()
