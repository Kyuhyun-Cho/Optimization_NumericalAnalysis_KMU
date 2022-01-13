import sys
import numpy as np


# update x by learning and gradient
def __update__(x, learning_rate, update_value):
    return x - learning_rate * update_value


# compute new gradient with momentum from previous moment and new gradient.
def __moment__(beta, gradient_moment, gradient):
    return beta * gradient_moment + gradient


def __weighted_average__(rho, gradient_weighted_average, gradient):
    return rho*gradient_weighted_average + (1-rho)*gradient


def __reciprocal_sqrt__(val):
    epsilon = sys.float_info.epsilon
    return 1 / np.sqrt(np.array(val + epsilon,dtype=np.float))


def __update_learning_rate__(learning_rate, beta1, beta2):
    return learning_rate * (np.sqrt(1 - beta2)/(1-beta1))