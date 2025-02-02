import numpy as np


class Perceptron:
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = np.zeros(self.input_size)
        self.bias = 0

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        activation = 1 / (1 + np.exp(-z))
        return 1 if activation >= 1 else 0

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            for input, target in zip(X, y):
                pred = self.predict(input)
                current_error = target - pred
                print(f"{epoch} error rate: {current_error}")
                self.weights = self.weights + learning_rate * current_error * input
                self.bias = self.bias + learning_rate * current_error


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

perceptron = Perceptron(input_size=2)
perceptron.train(X, y, learning_rate=0.1, epochs=1000)

new_data = np.array([[1, 0], [0, 1], [1, 1]])
predictions = [perceptron.predict(data) for data in new_data]
print(predictions)
