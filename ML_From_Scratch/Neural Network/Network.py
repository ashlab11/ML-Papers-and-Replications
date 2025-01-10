import numpy as np
from Layer_Dense import *
from Activations import *
from ErrorFunctions import *

class Network: 
    def __init__(self, layers = None, error_func = None) -> None:
        self.error_func = error_func
        self.errors = []
        if layers is None:
            self.layers = []
        else:
            self.layers = layers
    def add(self, layer):
        self.layers.append(layer)    
    def forward(self, inputs, y_true):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        self.errors = self.error_func.forward(inputs, y_true)
        return self.errors
    def backward(self):
        #We need to begin with the error function and the final layer, and then move backwards
        layer_grad = self.error_func.backward() #This is the derivative of the error function
        for layer in self.layers[::-1]:
            layer_grad = layer.backward(layer_grad) #Going backwards one by one!
    def train(self, X, y, epsilon = 0.01, epochs = 5, batch_frac = 0.1):
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_frac = batch_frac
        for _ in range(self.epochs):
            errors = []
            for idx in np.random.choice(range(len(X)), 100):
                X_row = X[idx].reshape(1, -1)
                y_row = y[idx]
                error = self.forward(X_row, y_row)
                errors.append(error)
                self.backward()
            print(f"Epoch {_ + 1}, Error: {np.mean(errors)}")
    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
            
        
