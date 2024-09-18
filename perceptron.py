import numpy as np
class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    def fit(self, X, y):
        """
        Train the Perceptron model.
        Args:
            X: numpy array of shape (n_samples, n_features) - Training data
            y: numpy array of shape (n_samples,) - Labels (+1 or -1)
        """
        n_samples, n_features = X.shape
        # I can Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
