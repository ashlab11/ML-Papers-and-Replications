import numpy as np
from Activations import *
import matplotlib.pyplot as plt
from Abstract_Classes import Layer

def conv2D(image, kernel, stride = 1, padding = 0):
    """Convolve an image with a kernel"""
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    output_height = (image_height - kernel_height + 2 * padding) // stride + 1
    output_width = (image_width - kernel_width + 2 * padding) // stride + 1
    output_array = np.zeros((output_height, output_width))
    
    for width in range(output_width):
        for height in range(output_height):
            pass
    pass

class Layer_Conv:
    def __init__(self, n_filters, filter_size, stride, activation, epsilon = 0.001) -> None:
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.stride = stride
        self.activation = activation
        self.epsilon = epsilon
        pass
    def forward(self, inputs):
        pass
    def backward(self, backward_grad):
        pass


#Convolution tests
image1 = [
    [1, 2, 3]
    [0, -1, 0], 
    [-1, 2, 0]
]
kernel1 = [
    [1, 0], 
    [0, 2]
]
output1 = [
    [-1, 2], 
    [4, -1]
]

print(conv2D(image1, kernel1))