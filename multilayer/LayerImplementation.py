from typing import Callable

import numpy as np
from AbstractClasses import Layer
import numpy as np
# shape in numpy
# z, y, x - depth, height, width respectively


# initialization
def glorot_uniform(shape):
    """Initialize weights with Glorot uniform distribution."""
    if len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape) == 3:
        receptive_field_size = np.prod(shape[:2])
        fan_in, fan_out = receptive_field_size * shape[2], shape[2]
    else:
        raise ValueError("Shape should be 2D or 3D.")

    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)


class ConvolutionalLayer(Layer):
    def backward(self):
        pass

    def __str__(self):
        return f'Conv(filter={self.num_filters}, kernel={self.kernel_size}, stride={self.stride}, padding={self.padding})'

    def __init__(self,
                 kernel_size: tuple = (3, 3),
                 num_filters: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 use_bias: bool = False,
                 initializer: Callable = glorot_uniform):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.filters = initializer((num_filters, *kernel_size))

        if use_bias:
            self.bias = np.random.randn(num_filters, 1, 1)
        else:
            self.bias = np.zeros((num_filters, 1, 1))

    def forward(self, input):
        """
           The forward computation for a convolution function

           Arguments:
           X -- output activations of the previous layer, numpy array of shape (height, width) assuming input channels = 1

           Returns:
           H -- conv output, numpy array of size (n_H, n_W)
           cache -- cache of values needed for conv_backward() function
        """

        # dimensions of filters aka z,y,x
        # also height and width of kernel should be the same, but i wont check for it
        num_filter, kernel_height, kernel_width = self.filters.shape

        padded_input = np.pad(input, self.padding, mode='constant')
        # print(f"padded input shape {padded_input.shape}")
        # Retrieving dimensions from X's shape
        input_height, input_width = padded_input.shape

        # Compute the output dimensions
        new_height = (input_height - kernel_height) // self.stride + 1
        new_width = (input_width - kernel_width) // self.stride + 1

        # Initialize the output H with zeros
        output_matrix = np.zeros((num_filter , new_height, new_width))
        # output_matrix = np.zeros((new_height, new_width))

        # Looping over vertical(h) and horizontal(w) axis of output volume
        for h in range(0, new_height):
            for w in range(0, new_width):
                input_slice = padded_input[h * self.stride: h * self.stride + kernel_height,
                        w * self.stride: w * self.stride + kernel_width]

                for z in range(self.num_filters):
                    output_matrix[z, h, w] = np.sum(input_slice * self.filters[z] + self.bias[z])

        # Saving information in 'cache' for backprop
        cache = (input, self.filters)

        return output_matrix, cache


if __name__ == '__main__':
    # Example usage
    # z, y, x = 5, 4, 3  # dimensions of the array
    #
    # array = glorot_uniform((z, y, x))
    # print(array)
    # print(array.shape)
    array = np.array([[1, 0, 0.5, 0.5],
                      [0, 0.5, 1, 0],
                      [0, 1, 0.5, 1],
                      [1, 0.5, 0.5, 1]])
    print(array.shape)
    convo_layer = ConvolutionalLayer(kernel_size=(2,2), num_filters=1, stride=1, padding=1, use_bias=False)
    print(convo_layer.filters)
    forward_pass = convo_layer.forward(array)[0]
    print(forward_pass[0])
    print(forward_pass[0].shape)
