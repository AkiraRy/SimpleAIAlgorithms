from typing import Callable

import numpy as np
from PIL.Image import Image

from AbstractClasses import Layer
import numpy as np
# shape in numpy
# z, y, x - depth, height, width respectively


def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out


def glorot_uniform(shape):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / (fan_in + fan_out))
    return np.random.uniform(low=-s, high=s, size=shape)



# initialization
# def glorot_uniform(shape):
#     """Initialize weights with Glorot uniform distribution."""
#     if len(shape) == 2:
#         fan_in, fan_out = shape[0], shape[1]
#     elif len(shape) == 3:
#         receptive_field_size = np.prod(shape[:2])
#         fan_in, fan_out = receptive_field_size * shape[2], shape[2]
#     else:
#         raise ValueError("Shape should be 2D or 3D.")
#
#     limit = np.sqrt(6 / (fan_in + fan_out))
#     return np.random.uniform(-limit, limit, size=shape)


class ConvolutionalLayer(Layer):
    def backward(self):
        pass

    def __str__(self):
        return f'Conv(filter={self.num_filters}, kernel={self.kernel_size}, stride={self.stride}, padding={self.padding})'

    def __init__(self,
                 kernel_size: tuple = 3,
                 num_filters: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 use_bias: bool = False,
                 initializer: Callable = glorot_uniform):

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = None
        self.weights_setup = False

    def forward(self, input):
        """
           The forward computation for a convolution function

           Arguments:
           X -- output activations of the previous layer, numpy array of shape (height, width) assuming input channels = 1
           Input should have those dimensions b, d, h, w = Batch number, depth, heigh. width
           Returns:
           H -- conv output, numpy array of size (n_H, n_W)
           cache -- cache of values needed for conv_backward() function
        """
        batch_number,  height, width, depth = input.shape
        output_dim_temp = height - self.kernel_size + 2*self.padding
        assert output_dim_temp % self.stride == 0, (f"In the convolutional layer, with stride {self.stride}"
                                                    f" can`t process whole image evenly.")

        output_dim = output_dim_temp/self.stride + 1
        output_depth = self.num_filters

        if not self.weights_setup:
            self.setup_weights(height, depth)
            self.weights_setup = True

        # dimensions of filters aka y,x,z
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
        output_matrix = np.zeros((num_filter, new_height, new_width))
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

    def setup_weights(self, input_dim, depth):
        self.weights_setup

def main():
    # Print the shape of the batch matrix
    # print("Shape of batch matrix:", batch_matrix.shape)

    # # Example usage
    # z, y, x = 5, 4, 3  # dimensions of the array
    # #
    # # array = glorot_uniform((z, y, x))
    # array = np.random.rand(2, z, y, x)
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


def main_test_initializer():
    num_filters = 32  # Number of filters (output channels)
    kernel_size = 3  # Kernel size (e.g., 3x3 kernel)
    num_channels = 3  # Number of input channels (e.g., RGB channels)

    # Constructing the shape tuple
    example_shape = (num_filters, kernel_size, kernel_size, num_channels)
    example_filter_weights  = glorot_uniform(shape=example_shape)

    print(example_filter_weights[0])
    print(example_filter_weights[0].shape)
    filter_to_show = example_filter_weights[0]

    min_value = np.min(filter_to_show)
    max_value = np.max(filter_to_show)
    normalized_weights = (filter_to_show - min_value) / (max_value - min_value) * 255
    normalized_weights = normalized_weights.astype(np.uint8)

    # Create a PIL image from the normalized weights
    pil_image = Image.fromarray(normalized_weights)

    # Show the PIL image
    pil_image.show()

if __name__ == '__main__':
    # main()
    main_test_initializer()
