'''
Модуль згорткової нейронної мережі дли обробки / прогнозування та екстраполяції часових рядів:
convolutional artificial neural network (CNN)


Класичний функціонал:
1. Підготовка даних:
    ділянка навчання / прогнозування;
    розбиття ділянки навчання на тренувальну / тестову вибірки в форматі навчальних пар;
    нормалізація даних
2. Навчання та контроль динаміки метрик;
3. Прогнозування;
4. Екстраполяція поза межами ділянки навчання.

Інноваційний функціонал:
1. Генерація опорної вибірки на ексттраполяцію з МНК та шумом.
2. Оптимізаційна модель вхідного прошарку.


Цікаві і корисні посилання:
LSTM модель LSTM (Long Short-Term Memory) / довга короткочасна память is a type of recurrent neural network (RNN)
https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
https://www.geeksforgeeks.org/deep-learning/tf-keras-layers-lstm-in-tensorflow/
https://www.geeksforgeeks.org/deep-learning/deep-learning-introduction-to-long-short-term-memory/
https://www.kaggle.com/code/hassanamin/time-series-analysis-using-lstm-keras
https://www.kaggle.com/code/kmkarakaya/keras-lstm-explained-in-details

Корисний приклад:
https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
https://keras.io/examples/timeseries/

Прогнозування подивитись на результат:
https://keras.io/examples/timeseries/timeseries_weather_forecasting/
https://www.tensorflow.org/tutorials/structured_data/time_series
https://www.tigerdata.com/blog/what-is-time-series-forecasting
https://www.tigerdata.com/blog/what-is-time-series-forecasting#neural-networks

Обзор методів екстраполяції в нейронках
https://www.tigerdata.com/blog/what-is-time-series-forecasting#neural-networks

Згорткова нейромережа для прогнозування
https://www.tigerdata.com/blog/what-is-time-series-forecasting#neural-networks
https://www.geeksforgeeks.org/deep-learning/multi-layer-perceptron-learning-in-tensorflow/

'''

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')




class MyModel(tf.keras.Model):
    '''
    Створення моделі нейромережі
    '''

    # def __init__(self, n_stocks, n_neurons, sigma):
    def __init__(self,  n_neurons, sigma):
        super(MyModel, self).__init__()
        weight_initializer = tf.keras.initializers.VarianceScaling(seed=7, scale=sigma, mode="fan_avg",
                                                                   distribution="uniform")
        bias_initializer = tf.zeros_initializer()
        self.hidden_layers = [tf.keras.layers.Dense(n_neurons[i], activation=tf.nn.relu,
                                                    kernel_initializer=weight_initializer,
                                                    bias_initializer=bias_initializer)
                              for i in range(len(n_neurons))]
        self.out = tf.keras.layers.Dense(1, kernel_initializer=weight_initializer,
                                         bias_initializer=bias_initializer)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.out(x)



def convolutional_neural_network_model(batch_size, epochs, real_data_in, data_predicted, n_neurons, k_expo, sigma):
    '''
    Головні процеси зі згортковою нейромережею
    '''



    def data_set_in(real_data_in, batch_size, k_expo):
        '''
        Формування потоку вхідних даних із залишком навчальної вибірки для прогнозу == екстраполяція
        '''

        # короткий датасет для навчання без прогнозу  k_expo
        n_real_data_in = len(real_data_in)
        real_data = np.zeros((n_real_data_in - k_expo))

        for f in range(n_real_data_in - k_expo):
            real_data[f] = real_data_in[f]

        # короткий датасет для навчання з batch_size
        m = len(real_data)
        n = batch_size
        train_data = np.zeros((m, n))
        test_data = np.zeros((m, n))

        for j in range(m):
            for i in range(n):
                train_data[j, i] = (real_data[j])
                test_data[j, i] = (real_data[j])

        return m, n, real_data, train_data, test_data



    def data_plot(data, text):
        '''
        Візуалізація потоку даних
        '''

        plt.figure(figsize=(15, 5))
        plt.plot(data)
        plt.title(text)
        plt.show()

        return


    def metrics_plot(history, title, text):
        '''
        Візуалізація динаміки метрик
        '''

        plt.figure(figsize=(15, 5))
        plt.plot(history.history[title])
        plt.title('metrics model')
        plt.ylabel(text)
        plt.xlabel('Epoch')
        plt.show()

        return



    def predict_plot(y_test, predicted, text):
        '''
        Візуалізація прогнозованих даних
        '''

        plt.figure(figsize=(15, 5))
        plt.plot(np.arange(y_test.shape[0]), y_test, color='blue', label='Real')
        plt.plot(np.arange(y_test.shape[0]), predicted, color='red', label='Predicted')
        plt.title(text)
        plt.legend()
        plt.show()

        return



    def data_norm(train_data, test_data):
        '''
        Нормалізація вхідних даних
        '''

        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_data_norm = scaler.fit_transform(train_data)
        test_data_norm = scaler.transform(test_data)


        return train_data_norm, test_data_norm, scaler



    def data_norm_opt(train_data, test_data, m, n):
        '''
        Нормалізація вхідних даних
        '''

        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_data_norm = scaler.fit_transform(train_data)
        test_data_norm = scaler.transform(test_data)


        # модиіфкація вхідного прохарку до оптимізаційного рівня

        def optimization_network(letter, delta):
            '''
            Оптимізація за Вороніним додана до згортки - найпростіший варіант
            '''

            x_letter = (1 / (1 / 0.9) * (1 - letter - delta))

            return x_letter

        def optimization_network_norm(letter, delta):
            '''
            Оптимізація за Вороніним додана до згортки - враховано критерії max, min -> min
            1 -> max = 1/(1+delta) -> min,
            0 -> min
            Користь в наближенні виходу до 1, 0. + швидкість навчання та ймовірність ідентифікації - покращено
            '''

            if (letter > 0.6):
                x_letter = (1 / (1 / 0.9) * (1 - (1 / (letter - delta))))
            else:
                x_letter = (1 / (1 / 0.9) * (1 - letter + delta))


            return x_letter

        # Модиіфікація вхідних даних за Вороніним
        for j in range(m):
            for i in range(n):
                train_data[j, i] = optimization_network_norm(train_data[j, i], 0)



        return train_data_norm, test_data_norm, scaler




    def dataset_formation(train_data, test_data):
        '''
        Формування навчального / тестового датасет із навчальних пар
        '''

        X_train, y_train = train_data[:, 1:], train_data[:, 0]
        X_test, y_test = test_data[:, 1:], test_data[:, 0]

        return X_train, y_train, X_test, y_test




    def network_training(model, X_train, y_train, epochs, batch_size):
        '''
        Тренування штучної нейронної мережі та оцінювання метрик
        Цікавий опис процесів: https://habr.com/ru/articles/482126/
        Функція втрат loss='mse' - мінімізується в процесі навчання найбільш вдала, оскільки передає природу часових рядів
        Метрики оцінки моделі - metrics:
        R-squared (R2) Score: (r2_score) - ймовірність апроксимації
        Mean Absolute Error (MAE) - середнє абсолютне відхилення
        Mean Squared Error (MSE) - середнє квадратичне відхилення
        loss - втрати - різниця вхід / вихід
        Оптимізатор та функція втрат - ті, що було в оригіналі
        optimizer = tf.keras.optimizers.Adam()
        model.compile(loss='MeanSquaredError', optimizer=optimizer)
        '''

        model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                      loss='mse',  # mean squared error
                      metrics=['r2_score', 'mae', 'mse'])

        # тренування моделі
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        # побудова динаміки метрк
        metrics_plot(history, 'r2_score', 'metrics: r2_score')
        metrics_plot(history, 'mae', 'metrics: mae')
        metrics_plot(history, 'mse', 'metrics: mse')
        metrics_plot(history, 'loss', 'metrics: loss')


        return model

    def network_predict(model_trained):
        '''
        Прогнозування часового ряду на ділянці навчання
        '''

        # прогнозування
        predicted = model_trained.predict(X_test)

        # візуалізація результатів
        predict_plot(y_test, predicted, 'Real vs Predicted normalized')


        return predicted



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


    def inverse_normalization(m, n, predicted, real_data):
        '''
        Ренормалізація - відновлення масштабу даних
        '''

        # повернення в оригінальну область параметрів
        predicted_inverse = np.zeros((m, n))
        for j in range(m):
            for i in range(n):
                predicted_inverse[j, i] = (predicted[j])


        # масштабуємо реальні значення
        predicted_n = scaler.inverse_transform(predicted_inverse)

        # повертаємо масштабовані до оригінальної шкали дані до необхідної структури
        for j in range(m):
            predicted[j] = predicted_n[j, 0]

        # обчислення метрик якості моделі на прогнозуванні
        metrics_predict(real_data, predicted, 'Метрики якості моделі відновлених даних')

        # візуалізація результатів
        predict_plot(real_data, predicted, 'Real vs Predicted original')


        return predicted



    def network_extrapolation(real_data_etalon, real_data_in, model, batch_size):
        '''
        Екстраполяція параметрів за межами інтервалу навчання

        По суті традиційно відбувається відтворення та екстраполяція даних в оригіналі.
        Для прогнозування потрібна вибірка, схожа на реальні дані.
        Формування опорної вибірки за межами інтервалу спостереження є окрема проблема.
        Тоді можливо брати МНК та генерувати опорну / тестову вибірку з моделі та інше.....
        Так це і реалізовано в коді.
        '''

        m = len(real_data_in)
        n = batch_size

        # --------- блок оригінальних даних ----------
        train_data = np.zeros((m, n))
        test_data = np.zeros((m, n))
        for j in range(m):
            for i in range(n):
                train_data[j, i] = (real_data_in[j])
                test_data[j, i] = (real_data_in[j])

        X_test, y_test = test_data[:, 1:], test_data[:, 0]

        # --------- блок нормованих даних ----------
        # закоментувати 2 рядка для прогнозування в оригінальних даних
        train_data_norm, test_data_norm, scaler = data_norm(train_data, test_data)
        X_train, y_train, X_test, y_test = dataset_formation(train_data_norm, test_data_norm)

        predicted = model.predict(X_test)

        # обчислення метрик якості моделі на прогнозуванні
        metrics_predict(y_test, predicted, 'Метрики якості моделі прогнозованих даних')

        # візуалізація результатів
        predict_plot(y_test, predicted, 'Real vs Predicted extrapolation')

        # Відновлення масштабу часового ряду на ділянці навчання
        predicted_original = inverse_normalization(m, n, predicted, real_data_in)

        predict_plot(real_data_etalon, predicted_original, 'Real vs Predicted original')



        return predicted_original



    '''
    Головні виклики для convolutional_neural_network_model
    
    '''

    # Потік вхідних даних із залишком навчальної вибірки для прогнозу == екстраполяція
    m, n, real_data, train_data, test_data = data_set_in(real_data_in, batch_size, k_expo)

    # Візуалізація
    data_plot(train_data, 'Вхідні параметри для тренування')
    data_plot(test_data, 'Вхідні параметри для тестування')

    # Нормалізація даних
    train_data_norm, test_data_norm, scaler = data_norm(train_data, test_data)
    # Модифікація вхідного прошарку оптимізаційною моделлю
    train_data_norm, test_data_norm, scaler = data_norm_opt(train_data, test_data, m, n)

    # Датасет з x та y - формування навчальних / тестових пар
    X_train, y_train, X_test, y_test = dataset_formation(train_data_norm, test_data_norm)

    # Створення екземпляру класу моделі нейромережі
    model = MyModel(n_neurons, sigma)

    # Тренування штучної нейронної мережі та оцінювання метрик
    model_trained = network_training(model, X_train, y_train, epochs, batch_size)

    # Прогнозування часового ряду на ділянці навчання
    predicted = network_predict(model_trained)

    # Відновлення масштабу часового ряду на ділянці навчання
    predicted_original = inverse_normalization(m, n, predicted, real_data)

    # Екстраполяція параметрів за межами інтервалу навчання
    predicted_extrapolation_original = network_extrapolation(real_data_in, data_predicted, model, batch_size)

    metrics_predict(real_data_in, predicted_extrapolation_original, 'Метрики якості екстраполяції оригінал даних')


    return predicted_original, predicted_extrapolation_original

