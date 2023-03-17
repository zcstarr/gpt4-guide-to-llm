### Exercise 1a: Linear Regression

**Overview:**

1. Linear regression is a method to model the relationship between a dependent variable (y) and one or more independent variables (x).
2. The goal is to find the best-fitting line (or hyperplane) that minimizes the sum of squared errors between the predicted and actual values.
3. Gradient descent is an optimization algorithm used to find the model parameters (weights and bias) that minimize the loss function (mean squared error).

**Resources:**

- [Linear Regression (Wikipedia)](https://en.wikipedia.org/wiki/Linear_regression)
- [Gradient Descent (Wikipedia)](https://en.wikipedia.org/wiki/Gradient_descent)
- [A Gentle Introduction to Linear Regression With Gradient Descent](https://towardsdatascience.com/a-gentle-introduction-to-linear-regression-with-gradient-descent-3b7465109d18)

**Context:**

1. `linear_regression(x, w, b)`: This function computes the predicted values (y_pred) given the input features (x), weights (w), and bias (b). Implement the linear equation y_pred = x \* w + b.
2. `mean_squared_error(y_true, y_pred)`: This function computes the mean squared error between the true values (y_true) and the predicted values (y_pred).
3. `compute_gradients(x, y_true, y_pred)`: This function computes the gradients of the loss function with respect to the weights (dw) and bias (db) using the input features (x), true values (y_true), and predicted values (y_pred).
4. `update_parameters(w, b, dw, db, learning_rate)`: This function updates the weights (w) and bias (b) using the gradients (dw, db) and the learning rate.

### Exercise 1b: Logistic Regression

**Overview:**

1. Logistic regression is a method for binary classification problems that models the probability of a data point belonging to a particular class.
2. The logistic function (or sigmoid) is used to convert the output of the linear regression into a probability value (between 0 and 1).
3. Gradient descent is used to optimize the model parameters (weights and bias) to minimize the loss function (binary cross-entropy).

**Resources:**

- [Logistic Regression (Wikipedia)](https://en.wikipedia.org/wiki/Logistic_regression)
- [Logistic Regression for Machine Learning](https://machinelearningmastery.com/logistic-regression-for-machine-learning/)

**Context:**

1. `sigmoid(z)`: This function computes the sigmoid function given the input (z). Implement the sigmoid function as 1 / (1 + exp(-z)).
2. `logistic_regression(x, w, b)`: This function computes the predicted probabilities given the input features (x), weights (w), and bias (b). Implement the logistic function applied to the linear equation (x \* w + b).
3. `binary_cross_entropy(y_true, y_pred)`: This function computes the binary cross-entropy loss between the true values (y_true) and the predicted probabilities (y_pred).
4. `compute_gradients(x, y_true, y_pred)`: This function computes the gradients of the loss function with respect to the weights (dw) and bias (db) using the input features (x), true values (y_true), and predicted probabilities (y_pred).
5. `update_parameters(w, b, dw, db, learning_rate)`: This function updates the weights (w) and bias (b) using the gradients (dw, db) and the learning rate.

### Exercise 1c: K-Means Clustering

**Overview:**

1. K-means clustering is an unsupervised learning algorithm that aims to partition a dataset into k clusters, where each data point belongs to the cluster with the nearest centroid.
2. The algorithm iteratively assigns data points to clusters based on their distance to the centroid and updates the centroids based on the new assignments until convergence or a maximum number of iterations is reached.
3. The quality of the clustering can be evaluated using metrics like the silhouette score, which measures the similarity of data points within the same cluster and their dissimilarity to data points in other clusters.

**Resources:**

- [K-means clustering (Wikipedia)](https://en.wikipedia.org/wiki/K-means_clustering)
- [K-means clustering: Algorithm, Applications, Evaluation Methods, and Drawbacks](https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a)

**Context:**

1. `initialize_centroids(X, k)`: This function initializes k centroids by randomly selecting k data points from the input data (X).
2. `assign_clusters(X, centroids)`: This function assigns each data point in X to the nearest centroid, forming clusters.
3. `update_centroids(X, cluster_assignments, k)`: This function computes the new centroids by calculating the mean of all data points assigned to each cluster.
4. `kmeans_clustering(X, k, max_iterations)`: This function performs the K-means clustering algorithm by iteratively assigning data points to clusters and updating centroids until convergence or the maximum number of iterations is reached.
