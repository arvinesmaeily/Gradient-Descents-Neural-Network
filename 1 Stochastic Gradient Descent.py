# import packages
import numpy as np

# define x and y that are correct input and output for training
x = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
y = np.array([[0, 1, 1, 0]]).T

# generate random weights
weights = np.random.random((3, 1))

# training process
for iteration in range(10000):
    for p in range(4):
        z = np.dot(x[p], weights)
        z = np.round(z, 2)
        sigmoid = 1/(1+np.exp(-z))
        error = y[p] - sigmoid
        sigmoidDerivative = sigmoid * (1 - sigmoid)
        delta = error * sigmoidDerivative
        dW = (delta * x[p])
        dW = np.array([dW]).T
        weights = weights + dW
print(weights)

# testing process
newZ = np.dot(np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]), weights)
ao = 1 / (1 + np.exp(-newZ))
print(ao)

