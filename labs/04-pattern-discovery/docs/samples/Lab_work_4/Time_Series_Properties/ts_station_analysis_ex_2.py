'''
Аналіз часового ряду на стаціонарність
https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html
https://www.geeksforgeeks.org/machine-learning/augmented-dickey-fuller-adf/
https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html

Стаціонарний часовий ряд (випадковий процес) має постійне значення статистичних властивостей мат.очикування
та дисперсію на обмеженому / необмеженому інтервалі часу (стаціонарність у вузькому / широкому сенсі):
параметрична надмірність даних;
часова надмірність даних.

'''



import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')




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


def dfuller_test_1(data_np):
    '''
    Оцінювання стаціонарності за Дікі-Фуллера (ADF)
    '''

    print("Results of Dickey-Fuller Test_1:")

    series = pd.Series(data_np)


    result = adfuller(series)

    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value}")

    rolling_mean = series.rolling(window=12).mean()
    rolling_std = series.rolling(window=12).std()

    plt.plot(series, label='Original')
    plt.plot(rolling_mean, label='Rolling Mean')
    plt.plot(rolling_std, label='Rolling Std')
    plt.legend()
    plt.title('Rolling Statistics')
    plt.show()

    return

def dfuller_test_2(data_np):

    print("Results of Dickey-Fuller Test_2:")

    timeseries = pd.Series(data_np)
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)

    return



def kpss_test(data_np):

    print("Results of KPSS Test:")

    timeseries = pd.Series(data_np)

    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)



if __name__ == '__main__':

    data_np_1 = file_parsing('data/Oschadbank (USD).xls', 'Купівля')
    data_np_2 = file_parsing('data/Oschadbank (USD).xls', 'Продаж')
    data_np_3 = file_parsing('data/Oschadbank (USD).xls', 'КурсНбу')

    dfuller_test_1(data_np_1)
    dfuller_test_1(data_np_1)
    kpss_test(data_np_1)


    '''
    Інтерпретація результатів dfuller_1:
    Статистика ADF: 
    Якщо вона значно нижча за критичні значення, відхилити нульове значення гыпотили.
    p-значення: Якщо p < 0,05, ряд, ймовірно, стаціонарний.
    Критичні значення: Використовуються для порівняння зі статистикою ADF.
    Оскільки -3,45 < -2,89 (рівень 5%), а p-значення < 0,05, ряд, ймовірно, є стаціонарним.
    
    Інтерпретація результатів dfuller_2
    Виходячи з рівня значущості 0,05 (0.25 - быльше за критичне) та p-значення ADF-тесту, нульову гіпотезу не можна відхилити. 
    Отже, ряд є нестаціонарним.
    
    Інтерпретація результатів kpss_test:
    Виходячи з рівня значущості 0,05 та p-значення тесту KPSS, 
    є докази для відхилення нульової гіпотези на користь альтернативної. 
    Отже, ряд є нестаціонарним згідно з тестом KPSS.
    
    '''