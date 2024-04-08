import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('US_population_dataset.xlsx')


df['age'].fillna(df['age'].mean(), inplace=True)
df['hours.per.week'].fillna(df['hours.per.week'].mean(), inplace=True)
data = df[['age', 'hours.per.week']].to_numpy()

def clusters(X, n_clusters, rseed=2):
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    while True:
        labels = np.argmin(np.linalg.norm(X - centers[:, np.newaxis], axis=2), 
                           axis=0)
        new_centers = np.array([X[labels == i].mean(0) 
                                for i in range(n_clusters)])
        if np.all(centers == new_centers):
            break
        centers = new_centers
    wcss = sum(np.min(np.linalg.norm(X - centers[:, np.newaxis], axis=2), axis=0))
    return centers, labels, wcss

max_clusters = 6
wcss_list = []
for n_clusters in range(1, max_clusters + 1):
    _, _, wcss = clusters(data, n_clusters)
    wcss_list.append(wcss)

plt.plot(range(1, max_clusters + 1), wcss_list, marker='o', linestyle='-')
plt.title('Elbow method')
plt.xlabel('Clusters')
plt.ylabel('WCSS')
plt.show()


n_clusters = 4
centers, labels, _ = clusters(data, n_clusters)

plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5, marker='D')
plt.xlabel('Age')
plt.ylabel('Hours per week')
plt.title('Clustering  Age / Hours per Week')
plt.show()