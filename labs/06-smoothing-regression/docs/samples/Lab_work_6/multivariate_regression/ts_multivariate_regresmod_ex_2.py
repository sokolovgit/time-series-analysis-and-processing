'''
Множинна регресія:
https://www.geeksforgeeks.org/machine-learning/ml-multiple-linear-regression-using-python/

Мета алгоритму — знайти найкраще рівняння прямої наближення, яке може передбачити значення на основі незалежних змінних.
Регресійна модель навчається з набору даних з відомими значеннями X та y та використовує його
для прогнозування значень y для невідомих X.

Приклад:
Датасет мыстить даны про житло в Каліфорнії:
медіанний дохід,
середня кількість кімнат
та цільову змінну – ціни на житло.

Візуалізація:
сині точки представляють фактичні ціни на житло на основі MedInc та AveRooms;
червона поверхня показує площину найкращого наближення, передбачену моделлю.
Ця візуалізація допомагаєзрозуміти, як ці дві характеристики впливають на прогнозовані ціни на житло.

'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
import warnings
warnings.filterwarnings('ignore')



def parsing():
    '''
    Парсинг даних
    '''

    california_housing = fetch_california_housing()

    X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    y = pd.Series(california_housing.target)

    return X, y




def split_train_test_sets( X, y):
    '''
    Утворення навчальної та тестової вибірок
    '''

    X = X[['MedInc', 'AveRooms']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test



def multiple_line_regression_mod(X_train, y_train, X_test, y_test):
    '''
    Модель множинної регресії
    '''

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_test['MedInc'], X_test['AveRooms'],
               y_test, color='blue', label='Actual Data')

    x1_range = np.linspace(X_test['MedInc'].min(), X_test['MedInc'].max(), 100)
    x2_range = np.linspace(X_test['AveRooms'].min(), X_test['AveRooms'].max(), 100)
    x1, x2 = np.meshgrid(x1_range, x2_range)

    z = model.predict(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape)

    ax.plot_surface(x1, x2, z, color='red', alpha=0.5, rstride=100, cstride=100)

    ax.set_xlabel('Median Income')
    ax.set_ylabel('Average Rooms')
    ax.set_zlabel('House Price')
    ax.set_title('Multiple Linear Regression Best Fit Line (3D)')

    plt.show()


    print('y_pred = ', y_pred)

    return y_pred




if __name__ == '__main__':


    X, y = parsing()

    X_train, y_train, X_test, y_test = split_train_test_sets(X, y)

    multiple_line_regression_mod(X_train, y_train, X_test, y_test)







