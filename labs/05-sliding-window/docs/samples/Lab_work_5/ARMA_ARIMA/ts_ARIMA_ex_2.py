'''
Алгоритм ARІMA - етапи
'''

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tools.eval_measures import rmse
import warnings
warnings.filterwarnings('ignore')



def parsing_plot(filename):
    '''
    Парсинг та візуалізатор
    '''

    # Read the AirPassengers dataset
    airline = pd.read_csv(filename, index_col='Month', parse_dates=True)

    # Print the first five rows of the dataset
    airline.head()

    return airline




def ts_decompose(airline):
    '''
    Декомпозиція
    '''

    # ETS Decomposition
    result = seasonal_decompose(airline['#Passengers'], model='multiplicative')

    airline['#Passengers'].plot()
    plt.show()

    # ETS plot
    result.plot()
    plt.show()

    return


def test_stationarity(series, title=''):
    '''
    Аналіз часового ряду на стаціонарність
    '''

    print(f"Results of ADF Test on {title}:")
    result = adfuller(series, autolag='AIC')
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] > 0.05:
        print("Non-stationary")
    else:
        print("stationary")
    for key, value in result[4].items():
        print(f"Critical Value ({key}): {value}")
    print("\n")

    return


def ACF_PACF_plot(airline):
    '''
    Аналіз автокореляції
    Вибір параметрів p і q з аналізу автокореляції та часткової автокореляції
    p і q визначають за викидами: вибираються останні значення горизонтальної осі,
    які мають точка виходить зарубіжних країн синьої області.
    Значення q виглядають за графіком ACF, a p - за графіком PACF.
    q = 0.5
    p = 0.6

    '''

    diff1 = airline['#Passengers'].diff()
    diff1.dropna(inplace=True)  # delete all rows with NaN value

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    half_size = len(diff1) // 2
    fig = sm.graphics.tsa.plot_acf(diff1, lags=half_size, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(diff1, lags=half_size, ax=ax2)
    plt.show()

    return



def SARIMA_predictions_and_KPI(airline):
    '''
    Прогнозування SARIMA та оцінювання якості моделі
    '''

    # Split data into train / test sets
    train = airline.iloc[:len(airline) - 12]
    test = airline.iloc[len(airline) - 12:]  # set one year(12 months) for testing

    # Fit a SARIMAX(0, 1, 1)x(2, 1, 1, 12) on the training set
    model = SARIMAX(train['#Passengers'], order=(0, 1, 1), seasonal_order=(2, 1, 1, 12))

    result = model.fit()
    result.summary()

    start = len(train)
    end = len(train) + len(test) - 1

    # Predictions for one-year against the test set
    predictions = result.predict(start, end,
                                 typ='levels').rename("Predictions")
    # plot predictions and actual values
    predictions.plot(legend=True)
    test['#Passengers'].plot(legend=True)
    plt.show()

    # Forecast for the next 3 years
    forecast = result.predict(start=len(airline),
                              end=(len(airline) - 1) + 3 * 12,
                              typ='levels').rename('Forecast')
    # Plot the forecast values
    airline['#Passengers'].plot(figsize=(12, 5), legend=True)
    forecast.plot(legend=True)
    plt.show()

    # Calculate root mean squared error
    rmse_out = rmse(test["#Passengers"], predictions)
    # Calculate mean squared error
    mean_squared_error_out = mean_squared_error(test["#Passengers"], predictions)
    # sklearn.metrics.r2_score
    r2_score_out = r2_score(test["#Passengers"], predictions)
    print('rmse_out = ', rmse_out)
    print('mean_squared_error_out = ', mean_squared_error_out)
    print('r2_score_out = ', r2_score_out)

    return




def ARIMA_predictions_and_KPI(airline):
    '''
    Прогнозування ARIMA та оцінювання якості моделі
    '''

    # Split data into train / test sets
    train = airline.iloc[:len(airline) - 12]
    test = airline.iloc[len(airline) - 12:]  # set one year(12 months) for testing

    # Fit a ARMA(0, 1, 1)x(2, 1, 1, 12) on the training set
    model = ARIMA(train['#Passengers'], order=(0, 1, 1), seasonal_order=(2, 1, 1, 12))

    result = model.fit()
    result.summary()

    start = len(train)
    end = len(train) + len(test) - 1

    # Predictions for one-year against the test set
    predictions = result.predict(start, end,
                                 typ='levels').rename("Predictions")
    # plot predictions and actual values
    predictions.plot(legend=True)
    test['#Passengers'].plot(legend=True)
    plt.show()

    # Forecast for the next 3 years
    forecast = result.predict(start=len(airline),
                              end=(len(airline) - 1) + 3 * 12,
                              typ='levels').rename('Forecast')
    # Plot the forecast values
    airline['#Passengers'].plot(figsize=(12, 5), legend=True)
    forecast.plot(legend=True)
    plt.show()

    # Calculate root mean squared error
    rmse_out = rmse(test["#Passengers"], predictions)
    # Calculate mean squared error
    mean_squared_error_out = mean_squared_error(test["#Passengers"], predictions)
    # sklearn.metrics.r2_score
    r2_score_out = r2_score(test["#Passengers"], predictions)
    print('rmse_out = ', rmse_out)
    print('mean_squared_error_out = ', mean_squared_error_out)
    print('r2_score_out = ', r2_score_out)

    return




# Парсинг
airline = parsing_plot('data/AirPassengers.csv')

# Декомпозиція
ts_decompose(airline)

# Аналіз на стаціонарність
test_stationarity(airline['#Passengers'], 'Passengers')

# Аналіз автокореляції
ACF_PACF_plot(airline)

# Прогнозування SARIMA
SARIMA_predictions_and_KPI(airline)

# Прогнозування ARIMA
ARIMA_predictions_and_KPI(airline)


