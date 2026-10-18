'''
Декомпозиція часових рядів / Time Series Decomposition з Loess(STL)
https://www.geeksforgeeks.org/data-analysis/seasonal-decomposition-of-time-series-by-loess-stl/
https://doc.arcgis.com/ru/insights/2022.1/analyze/stl.htm
STL — «Сезонна та трендова декомпозиція з використанням Loess».
Запропоновано R. B. Cleveland, Cleveland, McRae, & Terpenning (1990).
Алгоритм STL реалізує згладжування  часових ряди за допомогою LOESS у двох циклах:
внутрішній цикл виконує ітерацію між сезонним згладжуванням та згладжуванням тренду;
зовнішній цикл мінімізує вплив викидів.
У ході внутрішнього циклу сезонний компонент обчислюється першим та видаляється для обчислення компонента тренду.
Залишок обчислюється шляхом віднімання сезонного та трендового компононентів часових рядів

'''

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Завантаження dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')

# Plot the original time series
plt.figure(figsize=(7, 4))
plt.plot(df.index, df['Passengers'], label='Original Time Series', color='green')
plt.title('Air Passengers Dataset')
plt.xlabel('Year')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()


# Реалізація STL decomposition
stl = STL(df['Passengers'], seasonal=13, robust=True)
result = stl.fit()

# Візуалізація компонент декомпозиції
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 4))

ax1.plot(df.index, result.trend, label='Trend', color='red')
ax1.set_title('Trend Component')
ax2.plot(df.index, result.seasonal, label='Seasonal', color='blue')
ax2.set_title('Seasonal Component')
ax3.plot(df.index, result.resid, label='Residual')
ax3.set_title('Residual Component')
plt.tight_layout()
plt.show()


# Створення десезоналізованого часового ряду
deseasonalized_series = df['Passengers'] - result.seasonal

# visualize the comparative plot
plt.figure(figsize=(7, 4))
plt.plot(df.index, df['Passengers'], label='Original Time Series', color='blue')
plt.plot(df.index, deseasonalized_series, label='Deseasonalized Time Series', color='green')
plt.title('Original vs. Deseasonalized Time Series')
plt.xlabel('Year')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()


