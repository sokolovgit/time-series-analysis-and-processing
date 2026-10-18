'''
Експоненційне згладжування
https://www.geeksforgeeks.org/artificial-intelligence/exponential-smoothing-for-time-series-forecasting/
'''


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.api import Holt
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, r2_score
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



def Exponential_Smoothing(data):
    '''
    Просте або одноекспоненціальне згладжування

    '''

    model = SimpleExpSmoothing(data['#Passengers'])
    model_single_fit = model.fit()


    forecast_single = model_single_fit.forecast(len(data['#Passengers']))

    plt.plot(data, label='Original Data')
    plt.plot(model_single_fit.fittedvalues, label='Fitted Values')
    plt.plot(forecast_single, label='Forecast')
    plt.xlabel('Year')
    plt.ylabel('Number of Passengers')
    plt.title('Single Exponential Smoothing')
    plt.legend()
    plt.show()

    print('Просте або одноекспоненціальне згладжування')
    # Calculate mean squared error
    mean_squared_error_out = mean_squared_error(data, model_single_fit.fittedvalues)
    # sklearn.metrics.r2_score
    r2_score_out = r2_score(data, model_single_fit.fittedvalues)
    print('mean_squared_error_out = ', mean_squared_error_out)
    print('r2_score_out = ', r2_score_out, '\n')


    return



def Double_Exponential_Smoothing(data):
    '''
    Подвійне експоненціальне згладжування - Лінійна модель тренду Холта;

    '''


    model = Holt(data['#Passengers'])
    model_single_fit = model.fit()


    forecast_single = model_single_fit.forecast(len(data['#Passengers']))

    plt.plot(data, label='Original Data')
    plt.plot(model_single_fit.fittedvalues, label='Fitted Values')
    plt.plot(forecast_single, label='Forecast')
    plt.xlabel('Year')
    plt.ylabel('Number of Passengers')
    plt.title('Single Exponential Smoothing')
    plt.legend()
    plt.show()

    print('Подвійне експоненціальне згладжування - Лінійна модель тренду Холта')
    # Calculate mean squared error
    mean_squared_error_out = mean_squared_error(data, model_single_fit.fittedvalues)
    # sklearn.metrics.r2_score
    r2_score_out = r2_score(data, model_single_fit.fittedvalues)
    print('mean_squared_error_out = ', mean_squared_error_out)
    print('r2_score_out = ', r2_score_out, '\n')


    return





def Holt_Winter_Seasonal_Smoothing(data):
    '''
    Експоненціальне згладжування Холта-Вінтерса - Сезонна модель Холта-Вінтерса

    '''

    model = ExponentialSmoothing(
        data['#Passengers'],
        seasonal_periods=12,
        trend='add',
        seasonal='add')

    model_single_fit = model.fit()


    forecast_single = model_single_fit.forecast(len(data['#Passengers']))

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

    # Парсинг
    data = parsing_plot('data/AirPassengers.csv')

    # Просте або одноекспоненціальне згладжування
    Exponential_Smoothing(data)

    # Подвійне експоненціальне згладжування - Лінійна модель тренду Холта;
    Double_Exponential_Smoothing(data)

    # Експоненціальне згладжування Холта-Вінтерса - Сезонна модель Холта-Вінтерса
    Holt_Winter_Seasonal_Smoothing(data)




