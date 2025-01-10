import numpy as np
from Abstract_Classes import Error


class MSE(Error):
    def __init__(self) -> None:
        pass
    def forward(self, y_pred, y_true):
        self.loss = (y_pred - y_true) ** 2
        self.derivatives = 2 * (y_pred - y_true) # Derivative of the loss function
        return self.loss
    def backward(self):
        """Return the derivative of the loss function"""
        return self.derivatives
    
class CrossEntropy(Error):
    def __init__(self) -> None:
        pass
    def forward(self, y_pred, y_true):
        self.loss = -np.sum(y_true * np.log(y_pred))
        self.derivatives = -y_true / y_pred
        return self.loss
    def backward(self):
        return self.derivatives