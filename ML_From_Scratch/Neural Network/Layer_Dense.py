import numpy as np
from Activations import *
import matplotlib.pyplot as plt
from Abstract_Classes import Layer

    
class Layer_Dense(Layer):
    """Dense Layer"""
    def __init__(self, n_inputs, n_neurons, activation, epsilon = 0.001) -> None:
        """Initialize the layer with random weights and biases""" 
        self.epsilon = epsilon
        self.activation = activation       
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons) # Initializing weights with random values
        self.biases = np.zeros((1, n_neurons)) # Initializing biases with zeros
        #Initializing the derivatives
        self.weight_derivs = np.zeros((n_inputs, n_neurons))
        self.bias_derivs = np.zeros((1, n_neurons))
    def forward(self, inputs): 
        self.inputs = inputs # This will be used in the backward pass as well
        self.output_pre_activation = inputs @ self.weights + self.biases
        self.output = self.activation.forward(self.output_pre_activation)
        return (self.output)
    
    def backward(self, backward_grad):
        """We are only given $\delta^{l + 1}W^{l + 1}$, and have to calculate everything else!
        We return the delta of the layer"""
        activation_grad = self.activation.derivative(self.output_pre_activation) #Derivative of the activation function
        delta_layer = backward_grad * activation_grad #Component-wise derivative of the layer
        
        self.bias_derivs = delta_layer
        self.weight_derivs = self.inputs.T @ delta_layer
        
        #Updating weights and biases
        self.weights -= self.weight_derivs * self.epsilon
        self.biases -= self.bias_derivs * self.epsilon
        
        return delta_layer

class Layer_Input(Layer):
    def __init__(self, n_inputs) -> None:
        self.n_inputs = n_inputs
    def forward(self, inputs):
        self.inputs = inputs
        return (inputs)
    def backward(self, derivatives):
        return (derivatives)