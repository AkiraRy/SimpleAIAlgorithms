# Import Statements
from sklearn.model_selection import train_test_split

import abc
import numpy as np
import math
import os
import csv


# Model Component Definition
class ModelComponent(abc.ABC):
    @abc.abstractmethod
    def forward(self, u):
        pass

    @abc.abstractmethod
    def backward(self, del_v, lr):
        pass

    def update_learnable_parameters(self, del_w, del_b, lr):
        pass

    def save_learnable_parameters(self):
        pass

    def set_learnable_parameters(self):
        pass


# Convolution Layer Definition
class ConvolutionLayer(ModelComponent):
    def __init__(self, num_filters, kernel_size, stride=1, padding=0):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = None
        self.biases = None
        self.weights_matrix = None
        self.biases_vector = None
        self.u_pad = None

    def __str__(self):
        return f'Conv(filter={self.num_filters}, kernel={self.kernel_size}, stride={self.stride}, padding={self.padding})'

    def forward(self, u):
        num_samples = u.shape[0]
        input_dim = u.shape[1]
        output_dim = math.floor((input_dim - self.kernel_size + 2 * self.padding) / self.stride) + 1
        num_channels = u.shape[3]

        if self.weights is None:
            # ref: https://cs231n.github.io/neural-networks-2/#init
            # ref: https://stats.stackexchange.com/questions/198840/cnn-xavier-weight-initialization
            self.weights = np.random.randn(self.num_filters, self.kernel_size, self.kernel_size,
                                           num_channels) * math.sqrt(
                2 / (self.kernel_size * self.kernel_size * num_channels))
        if self.biases is None:
            # ref: https://cs231n.github.io/neural-networks-2/#init
            self.biases = np.zeros(self.num_filters)

        self.u_pad = np.pad(u, ((0,), (self.padding,), (self.padding,), (0,)), mode='constant')
        v = np.zeros((num_samples, output_dim, output_dim, self.num_filters))

        for k in range(num_samples):
            for l in range(self.num_filters):
                for i in range(output_dim):
                    for j in range(output_dim):
                        v[k, i, j, l] = np.sum(self.u_pad[k, i * self.stride: i * self.stride + self.kernel_size,
                                               j * self.stride: j * self.stride + self.kernel_size, :] * self.weights[
                                                   l]) + self.biases[l]

        return v

    def backward(self, del_v, lr):
        num_samples = del_v.shape[0]
        input_dim = del_v.shape[1]
        input_dim_pad = (input_dim - 1) * self.stride + 1
        output_dim = self.u_pad.shape[1] - 2 * self.padding
        num_channels = self.u_pad.shape[3]

        del_b = np.sum(del_v, axis=(0, 1, 2)) / num_samples
        del_v_sparse = np.zeros((num_samples, input_dim_pad, input_dim_pad, self.num_filters))
        del_v_sparse[:, :: self.stride, :: self.stride, :] = del_v
        weights_prime = np.rot90(np.transpose(self.weights, (3, 1, 2, 0)), 2, axes=(1, 2))
        del_w = np.zeros((self.num_filters, self.kernel_size, self.kernel_size, num_channels))

        for l in range(self.num_filters):
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    del_w[l, i, j, :] = np.mean(np.sum(
                        self.u_pad[:, i: i + input_dim_pad, j: j + input_dim_pad, :] * np.reshape(
                            del_v_sparse[:, :, :, l], del_v_sparse.shape[: 3] + (1,)), axis=(1, 2)), axis=0)

        del_u = np.zeros((num_samples, output_dim, output_dim, num_channels))
        del_v_sparse_pad = np.pad(del_v_sparse, (
        (0,), (self.kernel_size - 1 - self.padding,), (self.kernel_size - 1 - self.padding,), (0,)), mode='constant')

        for k in range(num_samples):
            for l in range(num_channels):
                for i in range(output_dim):
                    for j in range(output_dim):
                        del_u[k, i, j, l] = np.sum(
                            del_v_sparse_pad[k, i: i + self.kernel_size, j: j + self.kernel_size, :] * weights_prime[l])

        self.update_learnable_parameters(del_w, del_b, lr)
        return del_u

    def update_learnable_parameters(self, del_w, del_b, lr):
        self.weights = self.weights - lr * del_w
        self.biases = self.biases - lr * del_b

    def save_learnable_parameters(self):
        self.weights_matrix = np.copy(self.weights)
        self.biases_vector = np.copy(self.biases)

    def set_learnable_parameters(self):
        self.weights = self.weights if self.weights_matrix is None else np.copy(self.weights_matrix)
        self.biases = self.biases if self.biases_vector is None else np.copy(self.biases_vector)


np.random.seed(42)
input_data = np.random.randn(16, 16)  # Example 28x28 input image
input_data = input_data.reshape(1, 16, 16, 1)
print(input_data)
print(input_data.shape)
# conv_layer = ConvolutionLayer(num_filters=32, kernel_size=3, stride=1, padding=1)
# output_data = conv_layer.forward(input_data)
# print("Output shape:", output_data.shape)