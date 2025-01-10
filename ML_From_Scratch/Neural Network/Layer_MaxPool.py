import numpy as np
from Activations import *
import matplotlib.pyplot as plt
from Abstract_Classes import Layer

class Layer_Pool:
    def __init__(self, epsilon = 0.001) -> None:
        pass
    def forward(self, inputs):
        self.inputs = inputs
        pass
    def backward(self, backward_grad):
        pass
