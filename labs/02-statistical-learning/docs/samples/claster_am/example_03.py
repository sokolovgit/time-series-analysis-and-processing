
'''
Приклад грунтовної кластеризації

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


print("Завантаження та підготовка даних часового ряду EURO STOXX 50...")
df = pd.read_csv('wsj_sx5p_data_1988_2024.csv')
print(f"Розмір початкового датасету: {df.shape}")

df = df.dropna(subset=['Close'])
print(f"Розмір після видалення рядків з NaN: {df.shape}")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

print(f"Період даних: з {df['Date'].min().date()} по {df['Date'].max().date()}")

df['Returns'] = df['Close'].pct_change() * 100

df['Volatility_20d'] = df['Returns'].rolling(window=20).std()

df['MA_50'] = df['Close'].rolling(window=50).mean()

df['MA_200'] = df['Close'].rolling(window=200).mean()

df['MA_Ratio'] = (df['MA_50'] / df['MA_200'] - 1) * 100

delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI_14'] = 100 - (100 / (1 + rs))

df = df.dropna()
print(f"Розмір після створення технічних індикаторів і видалення NaN: {df.shape}")



def feature_engineering(df):
    """Створення ознак для кластеризації"""
    features = df[['Returns', 'Volatility_20d', 'MA_Ratio', 'RSI_14']]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, features



def kmeans_clustering(X, n_clusters=3):
    """Кластеризація методом K-means"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    silhouette_avg = silhouette_score(X, labels)
    
    return labels, kmeans.cluster_centers_, silhouette_avg



def hierarchical_clustering(X, n_clusters=3):
    """Ієрархічна кластеризація"""
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hierarchical.fit_predict(X)

    silhouette_avg = silhouette_score(X, labels)
    
    return labels, silhouette_avg



def svm_clustering(X, n_clusters=3):
    """Кластеризація за допомогою SVM (One-class SVM)"""

    models = []
    results = np.zeros((X.shape[0], n_clusters))

    for i in range(n_clusters):
        nu_value = (i + 1) / (n_clusters + 1)
        model = svm.OneClassSVM(nu=nu_value, kernel='rbf', gamma='auto')
        models.append(model)
        results[:, i] = model.fit(X).decision_function(X)
    

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(results)

    silhouette_avg = silhouette_score(X, labels)
    
    return labels, silhouette_avg



def knn_clustering(X, n_clusters=3):
    """Кластеризація на основі k-найближчих сусідів"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    initial_labels = kmeans.fit_predict(X)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, initial_labels)
    labels = knn.predict(X)

    silhouette_avg = silhouette_score(X, labels)
    
    return labels, silhouette_avg


X_scaled, features_df = feature_engineering(df)


print("\nВизначення оптимальної кількості кластерів...")
range_n_clusters = range(2, 10)
silhouette_scores = []

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, labels)
    silhouette_scores.append(silhouette_avg)
    print(f"Для n_clusters = {n_clusters}, Silhouette Score: {silhouette_avg:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_scores, 'o-', color='blue')
plt.xlabel('Кількість кластерів')
plt.ylabel('Silhouette Score')
plt.title('Визначення оптимальної кількості кластерів')
plt.grid(True)
plt.savefig('plot_clusters/optimal_clusters.png')


optimal_clusters = range_n_clusters[np.argmax(silhouette_scores)]
print(f"\nОптимальна кількість кластерів за Silhouette Score: {optimal_clusters}")


print("\nЗастосування методів кластеризації з оптимальною кількістю кластерів...")


kmeans_labels, kmeans_centers, kmeans_score = kmeans_clustering(X_scaled, optimal_clusters)
df['KMeans_Cluster'] = kmeans_labels
print(f"K-means силует-оцінка: {kmeans_score:.4f}")


hierarchical_labels, hierarchical_score = hierarchical_clustering(X_scaled, optimal_clusters)
df['Hierarchical_Cluster'] = hierarchical_labels
print(f"Ієрархічна кластеризація силует-оцінка: {hierarchical_score:.4f}")


svm_labels, svm_score = svm_clustering(X_scaled, optimal_clusters)
df['SVM_Cluster'] = svm_labels
print(f"SVM силует-оцінка: {svm_score:.4f}")


knn_labels, knn_score = knn_clustering(X_scaled, optimal_clusters)
df['KNN_Cluster'] = knn_labels
print(f"KNN силует-оцінка: {knn_score:.4f}")

methods = ['K-means', 'Ієрархічна кластеризація', 'SVM', 'KNN']
scores = [kmeans_score, hierarchical_score, svm_score, knn_score]
best_method_idx = np.argmax(scores)
best_method = methods[best_method_idx]
best_labels = [kmeans_labels, hierarchical_labels, svm_labels, knn_labels][best_method_idx]
print(f"\nНайкращий метод кластеризації: {best_method} з силует-оцінкою {scores[best_method_idx]:.4f}")


df['Best_Cluster'] = best_labels


print("\nВізуалізація результатів кластеризації...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels, cmap='viridis', 
                      alpha=0.7, s=50, edgecolors='w')
plt.colorbar(scatter, label='Кластер')
plt.xlabel('Головна компонента 1')
plt.ylabel('Головна компонента 2')
plt.title(f'Візуалізація кластерів ({best_method})')
plt.savefig('plot_clusters/clusters_pca.png')


print("\nАналіз характеристик кластерів:")
cluster_stats = df.groupby('Best_Cluster').agg({
    'Returns': 'mean',
    'Volatility_20d': 'mean',
    'MA_Ratio': 'mean',
    'RSI_14': 'mean',
    'Close': ['mean', 'min', 'max', 'std']
}).round(2)

print(cluster_stats)


plt.figure(figsize=(15, 8))
for cluster in df['Best_Cluster'].unique():
    cluster_data = df[df['Best_Cluster'] == cluster]
    plt.plot(cluster_data['Date'], cluster_data['Close'], 'o-', 
             markersize=4, label=f'Кластер {cluster}')

plt.xlabel('Дата')
plt.ylabel('Ціна закриття')
plt.title('Індекс EURO STOXX 50 з розділенням на кластери')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plot_clusters/stoxx50_clusters_timeline.png')

yearly_cluster_count = df.groupby([df['Date'].dt.year, 'Best_Cluster']).size().unstack().fillna(0)

plt.figure(figsize=(15, 8))
yearly_cluster_count.plot(kind='bar', stacked=True, ax=plt.gca())
plt.xlabel('Рік')
plt.ylabel('Кількість торгових днів')
plt.title('Розподіл кластерів за роками')
plt.grid(True, alpha=0.3)
plt.savefig('plot_clusters/clusters_by_year.png')