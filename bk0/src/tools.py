# tools.py

import numpy as np

def sigmoid(z):
    """sigmoid function"""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function"""
    return sigmoid(z)*(1-sigmoid(z))

# Quadratic Cost Function
class QuadraticCost:

    @staticmethod
    def fn(a, y):
        """
        Return the cost associated with an output "a" and desired output
        "y".
        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """
        Return the error delta from the output layer.
        """
        return (a-y) * sigmoid_prime(z)

# Cross-Entropy Cost Function
class CrossEntropyCost:

    @staticmethod
    def fn(a, y):
        """
        Return the cost associated with an output "a" and desired output
        "y".
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """
        Return the error delta from the output layer.
        """
        return (a-y)
