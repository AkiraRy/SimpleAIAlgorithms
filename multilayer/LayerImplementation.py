from typing import Callable
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


class ConvolutionalLayer(Layer):
    def backward(self):
        pass

    def __str__(self):
        return f'Conv(filter={self.num_filters}, kernel={self.kernel_size}, stride={self.stride}, padding={self.padding})'

    def __init__(self,
                 kernel_size: int = 3,
                 num_filters: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 use_bias: bool = False,
                 initializer: Callable = glorot_uniform):

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.bias = None
        self.weights = None
        self.weights_setup = False

    def forward(self, Input):
        """
           The forward computation for a convolution function

           Arguments:
           X -- output activations of the previous layer, numpy array of shape (height, width) assuming input channels = 1
           Input should have those dimensions b, d, h, w = Batch number, depth, heigh. width
           Returns:
           H -- conv output, numpy array of size (n_H, n_W)
           cache -- cache of values needed for conv_backward() function
        """
        batch_number,  height, width, input_depth = Input.shape
        output_dim_temp = height - self.kernel_size + 2*self.padding
        assert output_dim_temp % self.stride == 0, (f"In the convolutional layer, with stride {self.stride}"
                                                    f" can`t process whole image evenly.")

        output_dim = int(output_dim_temp/self.stride) + 1
        output_depth = self.num_filters

        if not self.weights_setup:
            self.setup_weights(input_depth)
            self.weights_setup = True

        padded_input = np.pad(Input, ((0,), (self.padding,), (self.padding,), (0,)), mode='constant')
        # shape is batch_number, y, x, depth

        output_matrix = np.zeros((batch_number, output_dim, output_dim, self.num_filters))

        for batch_index in range(batch_number):
            for height in range(output_dim):
                for width in range(output_dim):
                    height_from = height * self.stride
                    height_to = height_from + self.kernel_size
                    width_from = width * self.stride
                    width_to = width_from + self.kernel_size

                    image_slice = padded_input[batch_index, height_from:height_to, width_from: width_to, :]

                    for filter_index in range(self.num_filters):
                        output_matrix[batch_index, height, width, filter_index] = \
                            np.sum(image_slice * self.weights[filter_index]) + self.bias[filter_index]

        cache = (Input, self.weights, self.bias)

        return output_matrix, cache

    def setup_weights(self, depth):
        print((self.num_filters, self.kernel_size, self.kernel_size, depth))
        self.weights = glorot_uniform(shape=(self.num_filters, self.kernel_size, self.kernel_size, depth))

        if self.use_bias:
            self.bias = np.random.randn(self.num_filters)
        else:
            self.bias = np.zeros(self.num_filters)


class MaxPollingLayer(Layer):
    def __init__(self, kernel_size: int, stride: int, mode="max_pool"):
        self.index_matrix = None
        self.input_shape = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode

    def average_pool(self, Input):
        pass

    def max_pool(self, Input: np.ndarray):
        batch_number, height, width, input_depth = Input.shape
        output_dim_temp = height - self.kernel_size

        assert output_dim_temp % self.stride == 0, (f"In the convolutional layer, with stride {self.stride}"
                                                    f" can`t process whole image evenly.")

        output_dim = output_dim_temp // self.stride + 1

        output_matrix = np.zeros((batch_number, output_dim, output_dim, input_depth))
        self.index_matrix = np.zeros((batch_number, output_dim, output_dim, input_depth)).astype(np.int32)
        self.input_shape = Input.shape

        for batch_index in range(batch_number):
            for height in range(output_dim):
                for width in range(output_dim):
                    height_from = height * self.stride
                    height_to = height_from + self.kernel_size
                    width_from = width * self.stride
                    width_to = width_from + self.kernel_size

                    image_slice = Input[batch_index, height_from:height_to, width_from: width_to, :]
                    for d in range(input_depth):
                        output_matrix[batch_index, height, width, d] = np.max(image_slice[:, :, d])
                        self.index_matrix[batch_index, height, width, d] = np.argmax(image_slice[:, :, d])

        return output_matrix

    def forward(self, Input):
            if self.mode == "max_pool":
                return self.max_pool(Input)
            return self.average_pool(Input)

    def backward(self):
        pass



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
    example_filter_weights = glorot_uniform(shape=example_shape)

    print(example_filter_weights[0])
    print(example_filter_weights[0].shape)


def test():
    np.random.seed(42)

    # input_data = np.random.randn(28, 28)  # Example 28x28 input image
    # input_data = input_data.reshape(1, 28, 28, 1)
    # print(get_fans((32, 3, 3, 1)))
    # # print(input_data.shape)
    # conv_layer = ConvolutionalLayer(num_filters=32, kernel_size=3, stride=1, padding=1)
    # output_data = conv_layer.forward(input_data)[0]
    # # print(output_data[0])
    #
    # print("Output shape:", output_data.shape)


if __name__ == '__main__':
    # main()
    # main_test_initializer()
    test()