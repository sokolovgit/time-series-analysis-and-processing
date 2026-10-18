
'''
Приклад RRN: LSTM моделі
'''

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



# Функція для читання даних з Excel
def read_excel_data(File_name, Data_name):
    """
    Читає дані з файлу Excel, обробляє їх і підготовлює для аналізу.
    Видаляє нульові значення, якщо їх кількість менше 20%.

    :param File_name: Шлях до файлу Excel
    :param Data_name: Колонка для аналізу
    :return: Масив даних з обраної колонки
    """
    data = pd.read_excel(File_name)  # Зчитуємо дані з Excel файлу у DataFrame
    data.columns = data.columns.str.strip()  # Видаляємо зайві пробіли з назв стовпців

    data[Data_name] = pd.to_numeric(data[Data_name], errors='coerce')  # Перетворюємо стовпець в числовий тип
    data.dropna(inplace=True)  # Видаляємо рядки з NaN значеннями

    initial_iter = len(data[Data_name])  # Початкова кількість даних
    zeros_count = initial_iter - np.count_nonzero(data[Data_name])  # Кількість нулів
    print(f'Кількість нулів = {zeros_count}, у % = {round(100 * zeros_count / initial_iter, 3)}%')  # Кількість нулів та їх відсоток

    if zeros_count / initial_iter < 0.2:  # Якщо нулів менше 20%
        data = data[data[Data_name] != 0]  # Видаляємо рядки з нульовими значеннями
        print("Оскільки нульових значень менше 20% - їх було видалено")

        # графіки тренда, вимірів з нормальним шумом

    plt.clf()
    plt.plot(data[Data_name].values)
    plt.show()

    return data[Data_name].values  # Повертаємо масив значень з вказаної колонки



# Функція для створення послідовностей (вікон) для LSTM
def create_sequences(data, sequence_length):
    """
    Створює послідовності для навчання та тестування.

    :param data: Масив даних
    :param sequence_length: Довжина вікна
    :return: Послідовності X та y
    """
    sequences = []  # Список для послідовностей (X)
    targets = []  # Список для цільових значень (y)
    for i in range(len(data) - sequence_length):  # Ітерація по даних, створюючи вікна
        sequences.append(data[i:i + sequence_length])  # Додаємо вікно послідовностей
        targets.append(data[i + sequence_length])  # Додаємо цільове значення
    return np.array(sequences), np.array(targets)  # Повертаємо NumPy масиви




# Основна функція
def main(File_name, Data_name, sequence_length=10, batch_size=64, epochs=20):
    """
    Завантаження даних, створення послідовностей, створення моделі LSTM, навчання та тестування моделі.
    :param File_name: шлях до файлу Excel
    :param Data_name: Колонка для аналізу
    :param sequence_length: Довжина вікна
    :param batch_size: Розмір пакета
    :param epochs: Кількість епох навчання
    :return: None
    """
    # Завантаження та нормалізація даних
    data = read_excel_data(File_name, Data_name)  # Зчитуємо дані з Excel
    scaler = MinMaxScaler(feature_range=(0, 1))  # Створюємо об'єкт для масштабування даних в діапазон (0, 1)
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()  # Масштабуємо дані

    # Створення послідовностей
    X, y = create_sequences(data_scaled, sequence_length)  # Створюємо послідовності
    train_size = int(len(X) * 0.8)  # Визначаємо розмір тренувальної вибірки (80%)
    X_train, y_train = X[:train_size], y[:train_size]  # Тренувальні дані
    X_test, y_test = X[train_size:], y[train_size:]  # тестові дані

    # Перетворення формату для LSTM (додаємо розмірність для часових рядів)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Створення моделі LSTM
    model = tf.keras.Sequential([  # Створюємо послідовну модель
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(sequence_length, 1)),  # LSTM шар з 50 нейронами
        tf.keras.layers.Dense(1)  # Зв'язаний шар з 1 нейроном (для прогнозування одного значення)
    ])

    # # Створення моделі LSTM
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
    #                            input_shape=[None]),
    #     tf.keras.layers.LSTM(40, return_sequences=True),
    #     tf.keras.layers.LSTM(40),
    #     tf.keras.layers.Dense(1),
    # ])


    model.compile(optimizer='adam', loss='mse')  # Компіляція моделі з оптимізатором Adam та середньоквадратичною помилкою

    # Навчання моделі
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))  # Навчання моделі

    # Прогнозування
    predictions = model.predict(X_test)  # Отримуємо прогнози на тестових даних
    predictions_rescaled = scaler.inverse_transform(predictions)  # Масштабуємо прогнози
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))  # Масштабуємо реальні значення

    # Візуалізація результату
    plt.figure(figsize=(15, 5))  # Створюємо фігуру для графіка
    plt.plot(y_test_rescaled, label="Реальні дані")  # Графік реальних даних
    plt.plot(predictions_rescaled, label="Прогнозовані дані")  # Графік прогнозованих даних
    plt.legend()
    plt.title('Прогноз часових рядів')  # Додаємо заголовок
    plt.show()  # Показ графіку


# Головна частина програми
if __name__ == '__main__':
    File_name = "data\Oschadbank (USD).xls"
    Data_name = "Купівля"  # Можна змінити за необхідності
    main(File_name, Data_name)  # Виклик основної функції
