import numpy as np
import matplotlib.pyplot as plt
from Layer_Dense import *
from Activations import *
from ErrorFunctions import *
from Network import Network

np.random.seed(0)

def create_regression_data(n_points, n_features):
    X = np.random.rand(n_points, n_features)
    y = X[:, 0] * 5 + np.sign(X[:, 1]) + np.sin(X[:, 2]) + 1 + np.random.randn(n_points)
    return X, y


#We're using row terminology -- each layer is a row
X, y = create_regression_data(1000, 3)

#What we want the network to look like
network = Network(layers = [
    Layer_Input(3), #Input layer
    Layer_Dense(3, 5, Activation_ReLU()),
    Layer_Dense(5, 1, Activation_None()), #Output layer
], error_func = MSE())

network.train(X, y, epochs=100)
network.predict(X)
