'''
Лінійна регресія з statsmodels
https://www.geeksforgeeks.org/data-science/time-series-regression/
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



def parsing(filename):
    '''
    Парсинг даних
    '''

    df = pd.read_csv(filename)

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.sort_values('Date')

    cols_to_clean = ['Page.Loads', 'Unique.Visits',
                     'First.Time.Visits', 'Returning.Visits']
    for col in cols_to_clean:
        df[col] = df[col].str.replace(',', '').astype(int)

    df.set_index('Date', inplace=True)
    df.head()
    print('df = ', df)

    return df



def data_lag_variables(df):
    '''
    Створення зміщення у часі в даних: попереднє - наступне
    '''

    df['Lag1'] = df['Unique.Visits'].shift(1)
    df['Lag2'] = df['Unique.Visits'].shift(2)

    df = df.dropna()
    df[['Unique.Visits', 'Lag1', 'Lag2']].head()

    print('df_lag_variables = ', df)

    return df


def split_train_test_sets(df):
    '''
    Утворення навчальної та тестової вибірок
    '''

    X = df[['Lag1', 'Lag2']]
    y = df['Unique.Visits']

    split_point = int(len(df) * 0.8)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    X_train.head()
    X_test.head()

    return X_train, y_train, X_test, y_test



def line_regression_mod(X_train, y_train, X_test, y_test):
    '''
    Модель лінійної регресії
    '''

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)

    y_pred = model.predict(X_test)

    print(pd.DataFrame({
        'Actual': y_test.head().values,
        'Predicted': y_pred[:5]
    }))

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    metrics_predict(y_test, y_pred, 'metrics_predict')

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual', color='green')
    plt.plot(y_test.index, y_pred, label='Predicted', color='black')
    plt.title('Time Series Regression: Unique Visits')
    plt.xlabel('Date')
    plt.ylabel('Unique Visits')
    plt.legend()
    plt.tight_layout()
    plt.show()


    return


def metrics_predict(real_data, predicted, text):

    '''
    Оцінювання метрік результатів прогнозування з нейромережею
    '''

    # Обчислення метрик якості моделі на прогнозуванні
    mae = mean_absolute_error(real_data, predicted)
    mse = mean_squared_error(real_data, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(real_data, predicted)

    print('\n')
    print(text)
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R2) Score: {r2}")

    return


if __name__ == '__main__':


    df = parsing('data/daily-website-visitors.csv')

    df = data_lag_variables(df)

    X_train, y_train, X_test, y_test = split_train_test_sets(df)

    line_regression_mod(X_train, y_train, X_test, y_test)







