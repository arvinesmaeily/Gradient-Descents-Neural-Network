# import packages
import numpy as np

# define x and y that are correct input and output for training
x = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float128)
y = np.array([[0, 1, 1, 0]]).T

# initialize dWsum
dWsum = np.zeros((3, 1))

# generate random weights
weight = np.random.random((3, 1))

# training process
for iteration in range(10000):
    for p in range(4):
        z = np.dot(x[p], weight)
        z = np.round(z, 2)
        sigmoid = 1/(1+np.exp(-z))
        error = y[p] - sigmoid
        sigmoidD = sigmoid * (1 - sigmoid)
        delta = error * sigmoidD
        dW = (delta * x[p])
        dW = np.array([dW]).T
        dWsum = dWsum + dW
    dWavg = dWsum / 4
    weight = weight + dWavg
print(weight)

# testing process
newZ = np.dot(np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]), weight)
ao = 1 / (1 + np.exp(-newZ))
print(ao)
