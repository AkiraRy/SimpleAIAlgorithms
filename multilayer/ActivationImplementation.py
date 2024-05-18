import numpy as np
from AbstractClasses import Activation


class ReLU(Activation):
    def __init__(self):
        self.input_x = None

    def forward(self, x):
        self.input_x = x
        return np.maximum(0, x)

    def backward(self, dO):
        """
        Input:
        - dO: derivative output of previous layer

        returns
        - dx: Gradient with respect to x
        """
        dx = np.copy(self.input_x)
        dx[dx > 0] = 1
        dx[dx <= 0] = 0
        dx = dO * dx
        return dx


class SoftMax(Activation):
    def __init__(self):
        self.input_x = None


    def forward(self, x):
        self.input_x = x
        exponents = np.exp(x)
        output = exponents / np.sum(exponents, axis=0)
        return output


    def backward(self, dO):
        # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        return dO