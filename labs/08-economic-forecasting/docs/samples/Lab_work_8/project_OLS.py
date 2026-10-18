#----------- L13 Приклад застосування пакету статистичних обчислень  statsmodels ---------------

# https://www.statsmodels.org/stable/examples/notebooks/generated/ols.html

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

np.random.seed(9876789)
#---------------------------------- Приклад 1 лінійного МНК ------------------------------------
nsample = 100                    # формування масиву вхідних даних
x = np.linspace(0, 10, 100)
X = np.column_stack((x, x**2))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)

X = sm.add_constant(X)          # аргумент
y = np.dot(X, beta) + e         # функція

model = sm.OLS(y, X)            # виклик методу МНК
results = model.fit()           # визначення параметрів та статистичних характеристик МНК оцінок
print(results.summary())
print('Parameters: ', results.params)
print('R2: ', results.rsquared)

#--------------------- Приклад 2 нелінійний (але лінійний за параметрами) МНК ------------------

nsample = 50                   # формування масиву вхідних даних
sig = 0.5
x = np.linspace(0, 20, nsample)
X = np.column_stack((x, np.sin(x), (x-5)**2, np.ones(nsample)))
beta = [0.5, 0.5, -0.02, 5.]

y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)

res = sm.OLS(y, X).fit()       # виклик методу МНК
print(res.summary())           # визначення параметрів та статистичних характеристик МНК оцінок

print('Parameters: ', res.params)
print('Standard errors: ', res.bse)
print('Predicted values: ', res.predict())

prstd, iv_l, iv_u = wls_prediction_std(res)
                                # графічна візуалізація результатів розрахунків
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x, y, 'o', label="data")
ax.plot(x, y_true, 'b-', label="True")
ax.plot(x, res.fittedvalues, 'r--.', label="OLS")
ax.plot(x, iv_u, 'r--')
ax.plot(x, iv_l, 'r--')
ax.legend(loc='best')
plt.show()

