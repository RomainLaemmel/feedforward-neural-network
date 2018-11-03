import numpy as np
import neural_network

def main():
	# Load mnist dataset
	mnist = np.load('mnist_dataset.npz')
	# Create a three-layer network
	# The first layer (input layer) contains 784 neurons
	# The second layer (hidden layer) contains 200 neurons
	# The third layer (output layer) contains 10 neurons
	nn = neural_network.NeuralNetwork([784, 200, 10])
	nn.train(mnist["train_images"], mnist["train_labels"], 10)
	accuracy = nn.evaluate(mnist["test_images"], mnist["test_labels"])
	print(accuracy)

if __name__ == "__main__":
	main()