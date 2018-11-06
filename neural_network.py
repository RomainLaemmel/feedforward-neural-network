import numpy as np

class NeuralNetwork:
    """A class to perform the gradient descent learning algorithm using 
    backpropagation to calculate gradients and update network's weights"""

    def __init__(self, layers):
        self.layers = layers
        self.nb_layers = len(layers)
        self.weights = [np.random.randn(x, y) * 0.1 for x, y in zip(layers[1:], layers[:-1])]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def compute_outputs(self, inputs):
        layers_outputs = []
        for w in self.weights:
            layers_outputs.append(self.sigmoid(np.dot(inputs, w.T)))
            inputs = layers_outputs[-1]
        return layers_outputs

    def compute_errors(self, expected_output, layers_output):
        layers_error = []
        for l in range(self.nb_layers - 1, 0, -1):
            if l == self.nb_layers - 1:
                error = expected_output - layers_output[l - 1]
            else :
                error = np.dot(layers_error[0] * self.sigmoid_derivative(layers_output[l]), self.weights[l])
            layers_error.insert(0, error)
        return layers_error

    def adjust_weights(self, layers_input, layers_output, layers_error):
        for w in range(len(self.weights)):
            for i in range(len(self.weights[w])):
                adjustment = np.dot(layers_input[w], layers_error[w][i] * self.sigmoid_derivative(layers_output[w][i]))
                self.weights[w][i] += adjustment

    def train(self, network_inputs, expected_outputs, epochs):
        for e in range(epochs):
            for network_input, expected_output in zip(network_inputs, expected_outputs):
                layers_output = self.compute_outputs(network_input)
                layers_error = self.compute_errors(expected_output, layers_output)
                layers_input = layers_output[:-1]
                layers_input.insert(0, network_input)
                self.adjust_weights(layers_input, layers_output, layers_error)

    def evaluate(self, network_inputs, expected_outputs):
        accuracy = 0
        nb_tests = 0.0
        for network_input, expected_output in zip(network_inputs, expected_outputs):
            layers_output = self.compute_outputs(network_input)
            if np.argmax(layers_output[-1]) == np.argmax(expected_output):
                accuracy += 1
            nb_tests += 1
        accuracy = accuracy / nb_tests
        return accuracy