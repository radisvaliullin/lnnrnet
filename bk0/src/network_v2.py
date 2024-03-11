# network_v1.py
'''
code from chapter 3
it uses same stochastic gradient descent and backpropagation
it uses cross-entropy cost function (instead quadratic cost function)(solve saturation problem)
'''

import numpy as np
import random
from tools import sigmoid, sigmoid_prime, CrossEntropyCost
from helper.mnist_decode import vectoriz_result

class Network:

    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        sizes is list with size of each layer
        0 - layer is input
        1-(n-1) is hidden layers
        n - last layer is output
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # init weight randomly
        self.default_weight_init()
        # define cost function
        self.cost = cost

    def default_weight_init(self):
        """
        Initialize each weight using a Gaussian distribution. With mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.
        Initialize the biases using a Gaussian distribution. With mean 0
        and standard deviation 1.
        """
        # basises has only for hidden and out layers. 0 layer is input layer.
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        # weigths
        self.weights = [
            np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    
    def large_weight_init(self):
        """
        Initialize the weights and biases using a Gaussian distribution
        with mean 0 and standard deviation 1.
        This weight and bias initializer uses the same approach as in
        Chapter 1 (for purposes of comparison). It will usually be better
        to use the default weight initializer instead.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """feedforward, return output for layer, where input "a" is output of previous layer"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, train_data, epochs, batch_size, eta,
            lmbda = 0.0,
            eval_data=None,
            mon_eval_cost=False,
            mon_eval_accur=False,
            mon_train_cost=False,
            mon_train_accur=False):
        """
        SGD - stochastic gradient descent
        train_data - training data list of tuples (x, y), where x input sample, y expected output
        epochs - number of epochs
        batch size - size of batch (mini batch)
        eta - learning rate eta (n, neeeeet)
        lmbda (lambda) - regularization parameter
        eval_data (evaluation data) - used for evaluate of network after each epoch (if provides) (test data or validation data)
        mon_xxx - flags for monitoring cost or accuracy
        return mon values per epoch
        """
        if eval_data: n_eval = len(eval_data)
        n_train = len(train_data)
        eval_cost, eval_accur = [], []
        train_cost, train_accur = [], []
        for j in range(epochs):
            random.shuffle(train_data)
            batches = [train_data[k:k+batch_size] for k in range(0, n_train, batch_size)]
            for batch in batches:
                self.update_batch(batch, eta, lmbda, n_train)

            # print output
            print("epoch %s training complete" % j)
            if mon_train_cost:
                cost = self.total_cost(train_data, lmbda)
                train_cost.append(cost)
                print("cost on training data: {}".format(cost))
            if mon_train_accur:
                accur = self.accuracy(train_data, convert=True)
                train_accur.append(accur)
                print("accuracy on training data: {} / {}".format(accur, n_train))
            if mon_eval_cost:
                cost = self.total_cost(eval_data, lmbda, convert=True)
                eval_cost.append(cost)
                print("cost on evaluation data: {}".format(cost))
            if mon_eval_accur:
                accur = self.accuracy(eval_data)
                eval_accur.append(accur)
                print("accuracy on evaluation data: {} / {}".format(
                    self.accuracy(eval_data), n_eval))
            print
        return eval_cost, eval_accur, train_cost, train_accur

    def update_batch(self, batch, eta, lmbda, n):
        """
        Update network (weights and biases) by applying SGD and backpropagation to a single batch (mini batch).
        batch (mini batch) - list of tuples (x, y)
        eta - learning rate
        lmbda (lambda) - regularization parameter
        n - total size of the training data set
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # calculate delta and nabla for batch (see details of backprop)
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # update weights and biases based on nabla_w and nabla_b
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

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
        delta = (self.cost).delta(zs[-1], activations[-1], y)
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

    def accuracy(self, data, convert=False):
        """
        Return the number of inputs in "data" for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag "convert" should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data.
        """
        if convert:
            results = [
                (np.argmax(self.feedforward(x)), np.argmax(y))
                for (x, y) in data]
        else:
            results = [
                (np.argmax(self.feedforward(x)), y)
                for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """
        Return the total cost for the data set "data".
        The flag "convert" should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectoriz_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost
