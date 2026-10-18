'''
Дослідження часових рядів методами візуалізації: 3D візуалізація
OLAP stands for Online Analytical Processing
https://www.geeksforgeeks.org/data-analysis/data-cube-or-olap-approach-in-data-mining/
https://www.oreilly.com/library/view/practical-time-series/9781492041641/
https://www.geeksforgeeks.org/python/how-to-draw-2d-heatmap-using-matplotlib-in-python/

'''



import pandas as pd
import numpy as np
import pylab
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



    # 3d візуалізатор реальних даних
    data_np_1 = file_parsing('data/Oschadbank (USD).xls', 'Купівля')
    data_np_2 = file_parsing('data/Oschadbank (USD).xls', 'Продаж')
    data_np_3 = file_parsing('data/Oschadbank (USD).xls', 'КурсНбу')

    xg = np.zeros((len(data_np_1)))
    for i in range(len(data_np_1)):
        xg[i] = i

    fig = pylab.figure()
    ax = fig.add_subplot(projection='3d')
    clr = ['g', 'r', 'b']
    ax.bar(xg, data_np_1, 1, zdir='y', color=clr[0])
    ax.bar(xg, data_np_2, 2, zdir='y', color=clr[1])
    ax.bar(xg, data_np_3, 3, zdir='y', color=clr[2])
    ax.set_xlabel('X Time')
    ax.set_ylabel('Y Data')
    ax.set_zlabel('Z quantity')
    pylab.show()



    # 3d візуалізатор абстрактних даних
    # https://matplotlib.org/stable/gallery/mplot3d/hist3d.html#sphx-glr-gallery-mplot3d-hist3d-py
    np.random.seed(19680801)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x, y = np.random.rand(2, 100) * 4
    hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0, 4], [0, 4]])

    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    plt.show()



