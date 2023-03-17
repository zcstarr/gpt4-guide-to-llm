# linear_regression.py

import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic dataset
np.random.seed(42)
x = np.random.rand(100, 1)
y = 2 + 3 * x + np.random.randn(100, 1)

# Visualize the dataset
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data')
plt.show()


def linear_regression(x, w, b):
    # Implement the linear regression function here
    pass


def mean_squared_error(y_true, y_pred):
    # Implement the mean squared error function here
    pass


def compute_gradients(x, y_true, y_pred):
    # Compute the gradients of the loss function with respect to w and b
    pass


def update_parameters(w, b, dw, db, learning_rate):
    # Update the parameters w and b using the computed gradients and learning rate
    pass


# Hyperparameters
learning_rate = 0.1
num_iterations = 1000

# Initialize the parameters
w = np.random.randn(1)
b = np.zeros(1)

# Training loop
for i in range(num_iterations):
    # Compute the predictions using the current parameters
    y_pred = None  # Replace with the appropriate function call

    # Calculate the loss using the mean squared error function
    loss = None  # Replace with the appropriate function call

    # Compute the gradients
    dw, db = None  # Replace with the appropriate function call

    # Update the parameters
    w, b = None  # Replace with the appropriate function call

    # Print the loss for every 100 iterations
    if i % 100 == 0:
        print(f'Iteration {i}: Loss = {loss}')

# Generate predictions using the trained model
y_pred = None  # Replace with the appropriate function call

# Visualize the results
plt.scatter(x, y, label='True data')
plt.plot(x, y_pred, color='red', label='Predicted data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Results')
plt.legend()
plt.show()

# Desired results:
# 1. A decreasing loss during training: As the model learns, the loss should decrease over time,
#    indicating that the model is converging to a better solution.
# 2. A fitted line in the final plot: The red line in the final plot should closely follow the
#    underlying pattern of the data points. Since the synthetic dataset was generated with a linear
#    relationship (y = 2 + 3x + noise), the learned model should approximate this relationship.
#    The slope (w) should be close to 3, and the intercept (b) should be close to 2.
