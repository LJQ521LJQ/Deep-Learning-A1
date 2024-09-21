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
            # Training the model over multiple epochs
        for epoch in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.where(linear_output >= 0, 1, -1)

                # Update rule: Perceptron weight and bias update
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """
        Make predictions on new data.
        Args:
            X: numpy array of shape (n_samples, n_features) - Data to predict
        Returns:
            numpy array of shape (n_samples,) - Predicted labels (+1 or -1)
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)
