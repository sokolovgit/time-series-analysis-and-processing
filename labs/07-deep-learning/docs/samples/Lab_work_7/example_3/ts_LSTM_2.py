'''
Приклад LSTM моделі
'''

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
import math
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import warnings
warnings.filterwarnings('ignore')



### Функції для створення моделі та шуму:
def noise_normal(n, mu=0.0, sigma=1.0):
    return np.random.normal(mu, sigma, size=n)


def noise_uniform(n, low=-1.0, high=1.0):
    return np.random.uniform(low, high, size=n)


def model(a, b, minx, maxx, num):
    x = np.linspace(minx, maxx, num)
    y = a * x ** 2 + b * x
    return x, y


### Модель даних:
def create_model():
    ### Константи
    # Модель виду: y = a*x^2 + b*x
    a = 1  # параметр параболи
    b = 2  # параметр параболи
    minx = 0  # мінімальне значення горизонтальної вісі
    maxx = 10  # максимальне значення горизонтальної вісі
    num = 1000  # кількість точок

    # Для нормального шуму:
    mu = 0
    sigma = 5

    # Для рівномірного шуму:
    un_low = -10
    un_high = 10

    x, y = model(a, b, minx, maxx, num)  # Ідеальний тренд (пункт 2)
    nn = noise_normal(len(y), mu, sigma)  # Модель генерації випадкової величини (пукнт 1)
    un = noise_uniform(len(y), un_low, un_high)  # Модель генерації випадкової величини (пукнт 1)
    unn = nn + un  # uniform & normal noise
    y_unn = y + unn  # Адитивна модель - пункт 3
    return x, y_unn


### Функція для створення аномальних вимірів:
def add_anomaly(minv, maxv, step, x):
    y_anom = [0] * len(x)
    for i in range(0, len(x), step):
        y_anom[i] = random.randint(minv, maxv)

    return y_anom


### Створюємо модель з шумом:
x, y = create_model()

### Додаємо аномальні виміри:
y += add_anomaly(int(min(y)), int(max(y)), 100, x)

df = pd.DataFrame(data={'x': x, 'y': y})
### Візуалізуємо модель з аномальними вимірами:
sns.lineplot(data=df, x='x', y='y')
plt.show()

### Формування dataframe:
def find_anomalies(df, window_size=10, threshold=3):
    df['y_ma'] = df.y.rolling(window=window_size, center=True, min_periods=1).mean()
    df['y_diff'] = df.y - df.y_ma
    hist_data = df.y_diff.dropna()
    mu, sigma = norm.fit(hist_data)
    mask = ~df.y_diff.between(mu - threshold * sigma, mu + threshold * sigma) & df.y_diff.notna()
    df['is_anomaly'] = 0
    df.loc[mask, 'is_anomaly'] = 1
    return df[['x', 'y', 'is_anomaly']]


df = find_anomalies(df)
### Очищений датасет:
df = df[df.is_anomaly == 0]

### Візуалізація очищеного датасет:
sns.lineplot(data = df, x = 'x', y = 'y')
plt.show()


### Гіперзмінні

look_back = 10
n_future = 5

### Підготовка даних
data = df['y'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)


train_size = int(len(data_scaled) * 0.70)
train_data = data_scaled[0:train_size, :]
test_data = data_scaled[train_size:len(data_scaled), :]


def create_multistep_dataset(dataset, look_back, n_future):
    X = []
    Y = []

    for i in range(len(dataset) - look_back - n_future + 1):
        X.append(dataset[i: i + look_back, 0])

        Y.append(dataset[i + look_back: i + look_back + n_future, 0])
    return np.array(X), np.array(Y)

X_train, Y_train = create_multistep_dataset(train_data, look_back, n_future)
X_test, Y_test = create_multistep_dataset(test_data, look_back, n_future)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

### Модель LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(n_future))

model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

history = model.fit(
    X_train,
    Y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, Y_test),
    verbose=1
)

test_predict = model.predict(X_test)

test_predict = scaler.inverse_transform(test_predict)

y_test_orig = scaler.inverse_transform(Y_test)

plt.figure(figsize=(15, 7))
plt.plot(data, label='Оригінальні дані', color='blue')
for i in range(0, len(test_predict), 15):
    predict_indices = np.arange(start=train_size + i + look_back, stop=train_size + i + look_back + n_future, step=1)
    plt.plot(predict_indices, test_predict[i, :], color='red', marker='o', linestyle='--')


plt.title('Прогноз часового ряду за допомогою LSTM')
plt.xlabel('Часові кроки')
plt.ylabel('Значення')
plt.plot([], [], color='red', marker='o', linestyle='--', label=f'Прогноз на {n_future} кроків')
plt.legend()
plt.grid(True)
plt.show()


### метрики
mse = mean_squared_error(y_test_orig, test_predict)
rmse = math.sqrt(mse)

print(f"Помилка RMSE на тестових даних: {rmse:.2f}")