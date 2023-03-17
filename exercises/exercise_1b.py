# Exercise 1b: Logistic Regression
# In this exercise, we'll implement logistic regression from scratch
# using gradient descent optimization for binary classification.
# We'll use a synthetic dataset generated using scikit-learn's
# make_classification function.

# logistic_regression.py

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def sigmoid(z):
    # Implement the sigmoid function here
    pass


def logistic_regression(x, w, b):
    # Implement the logistic regression function here
    pass


def binary_cross_entropy(y_true, y_pred):
    # Implement the binary cross-entropy loss function here
    pass


def compute_gradients(x, y_true, y_pred):
    # Compute the gradients of the loss function with respect to w and b
    pass


def update_parameters(w, b, dw, db, learning_rate):
    # Update the parameters w and b using the computed gradients and learning rate
    pass


# Load and prepare synthetic dataset
data, target = make_classification(
    n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42)

# Desired results:
# 1. A decreasing loss during training: As the model learns, the loss should decrease over time,
#    indicating that the model is converging to a better solution.
# 2. A high accuracy on the test set: The trained model should have a high accuracy on the test set,
#    indicating that it generalizes well to new data.
