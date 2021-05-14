import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def create_sigmoid(x):
    a = []
    for item in x:
        a.append(sigmoid(item))
    return a


def der_sigmoid(x):
    a = []
    for item in x:
        a.append(sigmoid(item) * (1 - sigmoid(item)))
    return a


def relu(x):
    a = []
    for item in x:
        a.append(max(0, item))
    return a


def der_relu(x):
    a = []
    for item in x:
        if item <= 0:
            a.append(0)
        else:
            a.append(1)
    return a


def tanh(x):
    return (1 - math.exp(-x)) / (1 + math.exp(-x))


def create_tanh(x):
    a = []
    for item in x:
        a.append(tanh(item))
    return a


def der_tanh(x):
    a = []
    for item in x:
        a.append(1 - tanh(item)**2)
    return a


def leaky_relu(x, c=0.01):
    a = []
    for item in x:
        a.append(max(item * c, item))
    return a


def der_leaky_relu(x, c=0.1):
    a = []
    for item in x:
        if item <= 0:
            a.append(c)
        else:
            a.append(1)
    return a


import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10., 10., 0.1)
fig, ax = plt.subplots(2, 4)
plt.suptitle("Activation functions")
ax[0][0].plot(x, create_sigmoid(x))

ax[0][0].set_title("Sigmoid")

ax[0][1].plot(x, der_sigmoid(x))
ax[0][1].set_title("Sigmoid derivative")

ax[0][2].plot(x, relu(x))
ax[0][2].set_title("ReLU")
ax[0][3].plot(x, der_relu(x))
ax[0][3].set_title("ReLU derivative")

ax[1][0].plot(x, create_tanh(x))
ax[1][0].set_title("Tanh")
ax[1][1].plot(x, der_tanh(x))
ax[1][1].set_title("Tanh derivative")
ax[1][2].plot(x, leaky_relu(x))
ax[1][2].set_title("LeakyReLU")
ax[1][3].plot(x, der_leaky_relu(x))
ax[1][3].set_title("LeakyReLU derivative")

plt.show()
