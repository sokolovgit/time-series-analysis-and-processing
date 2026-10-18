'''
Алгоритм ARІMA - під  ключ
'''


import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
from statsmodels.graphics.tsaplots import plot_predict
import requests

# згладжування за ковзним середнім
def plot_moving_average(series, n, name):
    rolling_mean = series.rolling(window=n).mean()
    plt.figure(figsize=(15, 5))
    plt.title(f'Moving average\n window size = {n} for {name}')
    plt.plot(rolling_mean, c='orange', label='Rolling mean trend')
    plt.plot(series[n:], label='Actual values')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

# тест на стаціонарність ряду
def dickey_fuller_test(series):
    test = smt.adfuller(series, autolag='AIC')
    print('adf: ', test[0])
    print('p-value: ', test[1])
    print('Critical values: ', test[4])
    if test[0] > test[4]['5%']:
        print('Наявні одиничні корені, ряд не стаціонарний.')
    else:
        print('Одиничні корені відсутні, ряд є стаціонарним.')


if __name__=="__main__":
    # парсинг сайту НБУ за допомогою HTTP методу GET
    url = "https://bank.gov.ua/NBU_Exchange/exchange_site?json"
    params = {"start":"20231015",
               "end":"20231115",
               "valcode":"usd",
               "sort":"exchangedate",
              }
    response = requests.get(url, params=params)
    r = response.json()
    print(r)

    # перетворення даних в датафрейм
    kurs = pd.DataFrame(r)
    kurs = kurs[["exchangedate", "cc", 'rate']]
    print(kurs.head())
    kurs.columns = ["Date", "Digital code", "Value"]
    kurs["Date"] = pd.to_datetime(kurs["Date"], dayfirst=True)
    kurs = kurs.set_index("Date")
    print(kurs.head())

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(kurs['Value'])
    plt.show()

    plot_moving_average(kurs['Value'], 7, "UAH-USD")

    # візуалізація сезонної декомпозиції
    price_decomposition = smt.seasonal_decompose(kurs['Value'])
    fig = price_decomposition.plot()
    fig.set_size_inches(15, 10)
    plt.show()

    # тест на стаціонарність ряду
    dickey_fuller_test(kurs['Value'])
    kursDiff = kurs['Value'].diff(periods=1).dropna()
    # тест на стаціонарність різниці
    dickey_fuller_test(kursDiff)

    train_data = kurs['Value'][:-7]
    # побудова моделі
    model = smt.ARIMA(train_data, order=(1, 1, 1)).fit()
    print(model.summary())

    # передбачення і візуалізація й аналіз моделі
    pred = model.predict(kurs['Value'].index[-7], kurs['Value'].index[-1])
    fig, ax = plt.subplots(figsize=(15, 10))
    kurs['Value'][-20:].plot(ax=ax)
    ax.vlines(kurs['Value'].index[-7], 35, 38, linestyle='--', color='r', label='Start of forecast')
    ax = plot_predict(model, kurs['Value'].index[-7], kurs['Value'].index[-1], dynamic=True, plot_insample=False, ax=ax)
    plt.show()

    # прогноз
    forecasts = model.forecast(7)
    print(forecasts)