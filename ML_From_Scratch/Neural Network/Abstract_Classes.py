import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):
    """Abstract Activation Function"""
    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def derivative(self, inputs):
        pass

class Error(ABC):
    @abstractmethod
    def forward(self, y_pred, y_true):
        pass

    @abstractmethod
    def backward(self):
        pass
    

class Layer(ABC):
    """Key facts about a layer:
    1. It has a forward/backward method
    2. Weights and biases connect to the PREVIOUS layer"""
    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, pre_activation_grad):
        pass