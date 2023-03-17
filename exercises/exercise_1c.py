# Exercise 1c: K-Means Clustering
# In this exercise, we'll implement the k-means clustering algorithm from scratch.
# We'll use a synthetic 2D dataset generated using scikit-learn's make_blobs function.

# k_means_clustering.py

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def initialize_centroids(data, k):
    # Implement the initialization of the centroids here
    pass


def compute_distances(data, centroids):
    # Compute the distances between each data point and the centroids
    pass


def assign_clusters(distances):
    # Assign each data point to the closest centroid
    pass


def update_centroids(data, assignments, k):
    # Update the centroids based on the new cluster assignments
    pass


def k_means_clustering(data, k, max_iterations):
    # Implement the k-means clustering algorithm here
    pass


# Load and prepare synthetic dataset
data, _ = make_blobs(n_samples=500, centers=4,
                     cluster_std=1.0, random_state=42)

# Run k-means clustering
k = 4
max_iterations = 100
centroids, assignments = k_means_clustering(data, k, max_iterations)

# Evaluate clustering performance using silhouette score
score = silhouette_score(data, assignments)
print(f'Silhouette score: {score}')

# Visualize the clusters and centroids
plt.scatter(data[:, 0], data[:, 1], c=assignments,
            cmap='viridis', marker='o', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.show()

# Desired results:
# 1. Convergence of the centroids: The algorithm should converge, with the centroids
#    stabilizing after a certain number of iterations.
# 2. A high silhouette score: The resulting clusters should have a high silhouette score,
#    indicating that the data points within each cluster are close to each other and far
#    from the data points in other clusters.
# 3. A visualization of the clusters: A plot showing the data points assigned to different
#    clusters, with each cluster having a distinct color, and the centroids marked with red 'x' symbols.
