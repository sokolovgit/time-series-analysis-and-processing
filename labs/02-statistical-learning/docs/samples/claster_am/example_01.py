
'''
https://www.geeksforgeeks.org/introduction-to-anomaly-detection-with-python/
Тестування методів виявлення аномалій від бібліотеки PyOD

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.utils.data import generate_data
from pyod.models.pca import PCA


# Start by creating a dataset using generate_data() from pyod
X_train, y_train = generate_data(train_only=True)

# Create dataframe from Pandas using the generated data
df_train = pd.DataFrame(X_train)
df_train['y'] = y_train

# Display first few rows
df_train.head()

sns.scatterplot(x=0, y=1, hue='y', data=df_train, palette="hls", legend="full")
plt.title('Ground Truth')
plt.show()

# Create PCA model
clf = PCA()

# Trains PCA model
clf.fit(X_train)

# Store predictions for inlier and outlier in array as 0s and 1s
y_train_pred = clf.labels_
y_train_scores = clf.decision_scores_

ax = sns.scatterplot(x=0, y=1, hue=y_train_scores, data=df_train, palette="RdBu_r")

# Using legends, results look bit varied
legend_labels = [f"{score:.2f}" for score in np.unique(y_train_scores)]  # Format scores up to 2 decimal places
ax.legend(title="Anomaly Scores", labels=legend_labels)  # Create legend with title and labels
plt.title('Anomaly Scores by PCA')
plt.show()