'''
Приклад простої кластеризації

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

'''
https://pypi.org/project/yellowbrick/
'''
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import davies_bouldin_score


exchange_rate_df = pd.read_csv('exchange_rates.csv')
exchange_rate_df.date = pd.to_datetime(exchange_rate_df.date)
exchange_rate_df.head()


dateparse = lambda x: pd.to_datetime(x, dayfirst=True)
dd = pd.read_excel('Oschadbank (USD).xls', parse_dates=['Дата'], index_col='Дата', date_parser=dateparse)

print(dd)

index_1 ='Дата'
index_2 ='КурсНбу'
plt.title(index_2)
dd[index_2].plot()                   # стовпчик index з аргументом 0-n
plt.show()


plt.scatter(exchange_rate_df.date, exchange_rate_df.exchange_rate)
plt.show()


kmeans = KMeans(n_init='auto')
visualizer = KElbowVisualizer(kmeans, k=(2, 11), timings=False)
visualizer.fit(exchange_rate_df[['exchange_rate']])
visualizer.show()


visualizer = KElbowVisualizer(kmeans, k=(2, 11), metric='silhouette', timings=False)
visualizer.fit(exchange_rate_df[['exchange_rate']])
visualizer.show()




def get_kmeans_score(data, center):
    kmeans = KMeans(n_clusters=center, n_init='auto')
    model = kmeans.fit_predict(data)
    score = davies_bouldin_score(data, model)

    return score


scores = []

centers = list(range(2, 11))
for center in centers:
    scores.append(get_kmeans_score(exchange_rate_df[['exchange_rate']], center))

scores = np.array(scores)
plt.plot(centers, scores, linestyle='--', marker='o', color='b')
plt.axvline(centers[np.argmin(scores)], linestyle='--', color='black')
plt.xlabel('K')
plt.ylabel('Davies Bouldin score')
plt.title('Davies Bouldin score vs. K')
plt.show()


kmeans = KMeans(n_init='auto')
visualizer = KElbowVisualizer(kmeans, k=(2, 11), metric='calinski_harabasz', timings=False)
visualizer.fit(exchange_rate_df[['exchange_rate']])
visualizer.show()


kmeans = KMeans(n_clusters=3, n_init='auto')
clusters = kmeans.fit_predict(exchange_rate_df[['exchange_rate']])
cluster_colors = {0: 'y', 1: 'purple', 2: 'green'}
clusters = np.array(list(map(lambda x: cluster_colors[x], clusters)))
plt.scatter(*exchange_rate_df.columns, data=exchange_rate_df, c=clusters)
plt.show()
