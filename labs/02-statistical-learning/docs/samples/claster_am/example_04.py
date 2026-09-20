'''
Приклад ефективної кластеризації даних

'''


# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('googleplaystore.csv')

# Display basic information about the dataset
print("Dataset Info:")
print(data.info())
print("\nSample Data:")
print(data.head())

# Drop rows with missing values
data = data.dropna()

# Select relevant features for clustering
# 'Rating', 'Reviews', 'Installs', 'Price', 'Category'
data = data[['Rating', 'Reviews', 'Installs', 'Price', 'Category']]

# Encode categorical data
label_encoder = LabelEncoder()
data['Category'] = label_encoder.fit_transform(data['Category'])

# Convert 'Installs' and 'Price' columns to numeric
data['Installs'] = data['Installs'].str.replace(',', '').str.replace('+', '').astype(int)
data['Price'] = data['Price'].str.replace('$', '').astype(float)

# Standardize numerical features
scaler = StandardScaler()
data[['Rating', 'Reviews', 'Installs', 'Price']] = scaler.fit_transform(data[['Rating', 'Reviews', 'Installs', 'Price']])

# Display processed data
print("\nProcessed Data:")
print(data.head())

# Define the range for potential number of clusters
cluster_range = range(1, 11)
inertia = []

# Calculate inertia for each number of clusters to determine the optimal k
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    inertia.append(kmeans.inertia_)

# Plot the inertia values to observe the "elbow"
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, inertia, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()

# Apply k-means clustering with the chosen number of clusters (k = 4)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(data)

# Add cluster labels to the dataset
data['Cluster'] = kmeans.labels_

# Display a sample of the clustered data
print("\nClustered Data (sample):")
print(data.head())

# Display cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Reduce dimensionality to 2D for visualization using PCA
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data.drop('Cluster', axis=1))

# Plot the clusters
plt.figure(figsize=(10, 8))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=data['Cluster'], cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title("2D Visualization of Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.show()

# Calculate silhouette score to evaluate clustering quality
silhouette_avg = silhouette_score(data.drop('Cluster', axis=1), data['Cluster'])
print(f"\nSilhouette Score: {silhouette_avg:.3f}")

#Enhancements...

# Apply MinMaxScaler instead of StandardScaler
scaler = MinMaxScaler()
data[['Rating', 'Reviews', 'Installs', 'Price']] = scaler.fit_transform(data[['Rating', 'Reviews', 'Installs', 'Price']])

# Test with a higher number of clusters:  k = 5
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(data.drop('Cluster', axis=1))
data['Cluster'] = kmeans.labels_

# Calculate silhouette score to evaluate new clustering quality
silhouette_avg = silhouette_score(data.drop('Cluster', axis=1), data['Cluster'])
print(f"New Silhouette Score: {silhouette_avg:.3f}")

# Reduce dimensionality to 2D for visualization using PCA
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data.drop('Cluster', axis=1))

# Plot the new clusters
plt.figure(figsize=(10, 8))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=data['Cluster'], cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title("2D Visualization of Clusters with MinMax Scaling and k=5")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.show()

