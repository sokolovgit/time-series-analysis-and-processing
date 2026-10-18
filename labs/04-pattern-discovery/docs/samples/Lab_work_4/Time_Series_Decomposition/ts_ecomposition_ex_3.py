'''
Декомпозиція часових рядів / Time Series Decomposition
Від посібника з statsmodels: https://www.statsmodels.org/dev/examples/notebooks/generated/stl_decomposition.html

'''

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.datasets import elec_equip as ds
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast



'''
1. Завантаження даних для аналізу та їх візуалізація
'''
co2 = pd.read_csv(
    "https://raw.githubusercontent.com/statsmodels/smdatasets/refs/heads/main/data/stl-decomposition/co2.csv",
    parse_dates=True,
    index_col=0,
).iloc[:, 0]

stl = STL(co2, seasonal=13)
res = stl.fit()
fig = res.plot()
plt.show()



'''
2. Вмділення тренду через методи робастної адаптації (robust - надійний, адаптивний)
https://link.springer.com/article/10.1007/s10586-024-05061-7
https://scispace.com/pdf/training-robust-deep-models-for-time-series-domain-novel-2t7tzn1s.pdf
'''

elec_equip = ds.load().data.iloc[:, 0]

def add_stl_plot(fig, res, legend):
    """Add 3 plots from a second STL fit"""
    axs = fig.get_axes()
    comps = ["trend", "seasonal", "resid"]
    for ax, comp in zip(axs[1:], comps):
        series = getattr(res, comp)
        if comp == "resid":
            ax.plot(series, marker="o", linestyle="none")
        else:
            ax.plot(series)
            if comp == "trend":
                ax.legend(legend, frameon=False)

stl = STL(elec_equip, period=12, robust=True)
res_robust = stl.fit()
fig = res_robust.plot()
res_non_robust = STL(elec_equip, period=12, robust=False).fit()
add_stl_plot(fig, res_non_robust, ["Robust", "Non-robust"])
plt.show()



'''
3. LOESS degree - локальна поліноміальна регресія - для виділення тренду
https://www.statsdirect.com/help/nonparametric_methods/loess.htm
https://www.epa.gov/sites/default/files/2016-07/documents/loess-lowess.pdf
'''

lines = plt.plot(res_robust.weights, marker="o", linestyle="none")
ax = plt.gca()
plt.show()
xlim = ax.set_xlim(elec_equip.index[0], elec_equip.index[-1])



'''
4. Оптимізація обчислень інструментами statsmodels:
https://www.statsmodels.org/dev/examples/notebooks/generated/stl_decomposition.html
'''

stl = STL(
    elec_equip, period=12, seasonal_deg=0, trend_deg=0, low_pass_deg=0, robust=True
)
res_deg_0 = stl.fit()
fig = res_robust.plot()
add_stl_plot(fig, res_deg_0, ["Degree 1", "Degree 0"])
plt.show()


rs = np.random.RandomState(0xA4FD94BC)
tau = 2000
t = np.arange(tau)
period = int(0.05 * tau)
seasonal = period + ((period % 2) == 0)  # Ensure odd
e = 0.25 * rs.standard_normal(tau)
y = np.cos(t / tau * 2 * np.pi) + 0.25 * np.sin(t / period * 2 * np.pi) + e
plt.plot(y)
plt.title("Simulated Data")
plt.show()



mod = STL(y, period=period, seasonal=seasonal)
res = mod.fit()
fig = res.plot(observed=False, resid=False)
plt.show()


low_pass_jump = seasonal_jump = int(0.15 * (period + 1))
trend_jump = int(0.15 * 1.5 * (period + 1))
mod = STL(
    y,
    period=period,
    seasonal=seasonal,
    seasonal_jump=seasonal_jump,
    trend_jump=trend_jump,
    low_pass_jump=low_pass_jump,
)

res = mod.fit()
fig = res.plot(observed=False, resid=False)
plt.show()



'''
Сумісне використання STL та ARIMA для прогнозування
Тобто розкладання на компоненти (зокрема тренд) допомогає адаптувати модель прогнозу
'''


elec_equip.index.freq = elec_equip.index.inferred_freq
stlf = STLForecast(elec_equip, ARIMA, model_kwargs=dict(order=(1, 1, 0), trend="t"))
stlf_res = stlf.fit()

forecast = stlf_res.forecast(24)
plt.plot(elec_equip)
plt.plot(forecast)
plt.show()
