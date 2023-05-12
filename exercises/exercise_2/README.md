### Exercise 2a: Implement a Simple Feedforward Neural Network (Multi-Layer Perceptron) for Classifying the MNIST Dataset

**Overview:**

1. A feedforward neural network, or multi-layer perceptron (MLP), is a type of artificial neural network that consists of an input layer, one or more hidden layers, and an output layer.
2. Each layer consists of multiple neurons, and the neurons in one layer are connected to the neurons in the next layer through weighted connections.
3. The network learns by adjusting these weights and biases during training to minimize the loss function.
4. In this exercise, you will implement a simple MLP to classify handwritten digits from the MNIST dataset.

**Resources:**

- [Feedforward Neural Networks (Wikipedia)](https://en.wikipedia.org/wiki/Feedforward_neural_network)
- [MNIST dataset (Wikipedia)](https://en.wikipedia.org/wiki/MNIST_database)
- [A Beginner's Guide to Neural Networks and Deep Learning](https://skymind.ai/wiki/neural-network)

**Context:**

1. Load the MNIST dataset and preprocess it by normalizing the pixel values and one-hot encoding the labels.
2. Implement the forward pass of the MLP, which includes computing the weighted sum of the inputs for each neuron, applying the activation function (e.g., ReLU or sigmoid), and repeating this process for each layer.
3. Implement the backward pass of the MLP, which involves computing the gradients of the loss function with respect to the weights and biases using backpropagation.
4. Implement the weight and bias update step using gradient descent or a more advanced optimization algorithm (e.g., Adam).
5. Train the MLP on the MNIST dataset and evaluate its performance on a test set.

**Expectations:**

- Implement a simple feedforward neural network (MLP) with at least one hidden layer.
- Successfully train the network to classify handwritten digits from the MNIST dataset.
- Achieve a reasonable level of accuracy (e.g., greater than 90%) on the test set.
- Demonstrate an understanding of the forward and backward pass, activation functions, and optimization algorithms used in training the network.
