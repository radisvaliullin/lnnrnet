# network_v1.py
'''
code from chapter 1 and 2
fast matrix-based algorithm implementing stochastic gradient descent and backpropagation
used cost function is quadratic cost function
'''

import numpy as np
import random
from tools import sigmoid, sigmoid_prime

class Network:

    def __init__(self, sizes):
        """
        sizes is list with size of each layer
        0 - layer is input
        1-(n-1) is hidden layers
        n - last layer is output
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # basises has only for hidden and out layers. 0 layer is input layer.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # weigths
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """feedforward, return output for layer, where a is output of previous layer"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, train_data, epochs, batch_size, eta, test_data=None):
        """
        SGD - stochastic gradient descent
        train_data - training data list of tuples (x, y), where x input sample, y expected output
        epochs - number of epochs
        batch size - size of batch (mini batch)
        eta - learning rate eta (n, neeeeet)
        test_data - used for evaluate of network after each epoch (if provides)
        """
        if test_data: n_test = len(test_data)
        n = len(train_data)
        for j in range(epochs):
            random.shuffle(train_data)
            batches = [train_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                self.update_batch(batch, eta)
            if test_data:
                print("epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("epoch {0} complete".format(j))

    def update_batch(self, batch, eta):
        """
        Update network (weights and biases) by applying SGD and backpropagation to a single batch (mini batch).
        batch (mini batch) - list of tuples (x, y)
        eta - learning rate
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # calculate delta and nabla for batch (see details of backprop)
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # update weights and biases based on nabla_w and nabla_b
        self.weights = [w-(eta/len(batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Calculates a value (nabla_b, nabla_w) the gradient for the cost function C_x.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        #
        # feedforward
        #
        activation = x
         # list of all the activations, layer by layer
        activations = [x]
        # list of all the z vectors, layer by layer
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        # delta - dC/dz is error of neuron output
        # delta is vector (errors of neurons in layer)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        # nabla b - dC/db
        nabla_b[-1] = delta
        # nabla w - dC/dw
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # calculate delta and nabla for each layer
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        # return calculated gradients nabla
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural
        network outputs the correct result.
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives partial C_x / partial a for the output activations.
        """
        return (output_activations-y)
