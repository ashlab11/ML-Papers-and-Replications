import numpy as np
from Abstract_Classes import Activation

class Activation_ReLU(Activation):
    """Rectified Linear Unit Activation Function"""
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return (self.output)
    def derivative(self, inputs):
        self.derivatives = np.where(inputs <= 0, 0, 1)
        return (self.derivatives)
        
class Activation_Sigmoid(Activation):
    """Sigmoid Activation Function"""
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output
    def derivative(self, inputs):
        forward = self.forward(inputs)
        self.derivatives = forward * (1 - forward)
        self.output = self.derivatives
    
    
class Activation_None(Activation):
    """No Activation Function"""
    def forward(self, inputs):
        self.derivatives = np.ones_like(inputs)
        return inputs
    def derivative(self, inputs):
        return self.derivatives