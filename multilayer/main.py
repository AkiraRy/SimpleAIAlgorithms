import argparse
import csv
import math
import random
from typing import Union

import numpy as np

from AbstractClasses import Layer, Dataset, Activation
from LayerImplementation import ConvolutionalLayer, SoftMax, PollingLayer, Dense, Flatten
from ActivationImplementation import ReLU
from OptiomizerImplementation import SGD
from LossImplementation import CrossEntropyLoss
import matplotlib.pyplot as plt

from multilayer.data_reader import DatasetImages

optimizers = {
    "SGD": SGD,
}
losses = {
    'cross_entropy': CrossEntropyLoss
}


class Sequential:
    def __init__(self, ):
        self.layers = list()
        self.data_reader = None
        self.optimizer = None
        self.loss = None

    def add(self, item: (Layer, Activation)) -> None:
        if isinstance(item, (Layer, Activation)):
            self.layers.append(item)
        else:
            raise TypeError(f"Only instances of classes that extend Layer or Activation can be added, "
                            f"but `{type(item).__name__}` was provided.")

    def get(self, index: int) -> Layer:
        """
        Negative indexes like -1 will return last layer, take this into consideration
        :param index: of Layer in the Sequential model
        :return: Layer
        """
        if index < len(self.layers):
            return self.layers[index]
        else:
            raise IndexError("Index out of range")

    def compile(self, optimizer: str = 'SGD', loss='cross_entropy', metrics='accuracy'):
        self.optimizer = optimizers.get(optimizer)
        self.loss = losses.get(loss)

    def fit(self, data_reader: Dataset, epochs=100, batch_size=32):
        pass

    def evaluate(self):
        pass

    def single_data_forwrd(self, Input):
        output = Input
        out_list = []
        counter = 1
        for layer in self.layers:
            output = layer.forward(output)
            out_list.append(output)
            print(f"{counter} {output.shape = }")
            counter += 1
        return output, out_list

    def get_info(self):
        info = ''
        for layer in self.layers:
            info += layer.__str__()
            info += '\n'
        return info


def plot_layers(output):
    def plot_feature_maps(feature_maps, title):
        num_filters = feature_maps.shape[2]
        fig, axes = plt.subplots(1, num_filters, figsize=(num_filters * 2, 2))
        fig.suptitle(title)

        if num_filters == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.imshow(feature_maps[:, :, i], cmap='viridis')
            ax.axis('off')
        plt.show()

    # Plot the outputs
    plot_feature_maps(output[0][0], "Output of Conv Layer 1")
    plot_feature_maps(output[1][0], "Output of ReLU 1")
    plot_feature_maps(output[2][0], "Output of Pool 1")
    plot_feature_maps(output[3][0], "Output of Conv Layer 2")
    plot_feature_maps(output[4][0], "Output of ReLU 2")
    plot_feature_maps(output[5][0], "Output of Pool 2")
    plot_feature_maps(output[6][0], "Output of Conv Layer 3")

    # Since output7, output8, and output9 are flattened or 1D, we plot them directly
    plt.figure(figsize=(12, 4))
    plt.plot(output[7])
    plt.title("Output of Flatten Layer")
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(output[8])
    plt.title("Output of Fully Connected Layer")
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(output[9])
    plt.title("Output of SoftMax Layer")
    plt.show()


def test():
    model_param = {
        'l1': (ConvolutionalLayer, (5, 6, 1, 2)),
        'l2': (ReLU, ()),
        'l3': (PollingLayer, (2, 2)),
        "l4": (ConvolutionalLayer, (5, 12, 1, 0)),
        'l5': (ReLU, ()),
        'l6': (PollingLayer, (2, 2)),
        'l7': (ConvolutionalLayer, (5, 100, 1, 0)),
        'l8': (Flatten, ()),
        # 'l9': (Dense, (10,)),
        'l9': (Dense, (10, )),
        'l10': (SoftMax, ())
    }
    np.random.seed(42)

    model = Sequential()

    for key, value in model_param.items():
        model.add(value[0](*value[1]))

    my_generator = DatasetImages("dataset", 0.2, batch_size=2)
    getter = my_generator.get_next_data_row()
    batch = [next(getter)]

    batch_number = 0
    letter = 0
    data_img = 1

    image_data = batch[batch_number][data_img][0]
    image_data_shape = image_data.shape
    image_data_1 = image_data.reshape(1,image_data_shape[1], image_data_shape[0], 1)

    image_data2 = batch[batch_number][data_img][1]
    image_data_shape2 = image_data2.shape
    image_data_2 = image_data2.reshape(1,image_data_shape2[1], image_data_shape2[0], 1)
    batched_array = np.concatenate((image_data_1, image_data_2), axis=0)



    # batched_array = np.concatenate((array, array2), axis=0)

    output = model.single_data_forwrd(batched_array)
    print(output[0].shape)
    print(output[0])
    print(np.sum(output[0]))
    # print(model.get_info())

    # plot_layers([output, output1, output2, output3, output4, output5, output6, output7, output8, output9])
    # plot_layers(output[1])

# def default(args):
#     x_train, x_test, y_train, y_test = train_test_split(args.dataset_path, test_size=0.2)
#     n_epochs = args.e
#     learning_rate = args.alpha
#     value_map = convert_target_to_number(set(y_train))
#     nnetwork = NeuralNetwork(Layer(len(value_map), 26, linear), n_epochs, value_map, learning_rate)
#     accuracies = nnetwork.fit(x_train,y_train, x_test, y_test)
#     print(f"Accuracy of the model is: {nnetwork.test_accuracy()*100:.2f}%")
#     plt.plot(range(1, n_epochs + 1), accuracies)
#     plt.xlabel("Epochs")
#     plt.ylabel("Accuracy")
#     plt.title("Accuracy per epoch")
#     plt.show()
#
#
# def iterate(args):
#     x_train, x_test, y_train, y_test = train_test_split(args.dataset_path, test_size=0.2)
#     n_epochs = args.e
#     learning_rate = args.alpha
#     value_map = convert_target_to_number(set(y_train))
#     nnetwork = NeuralNetwork(Layer(len(value_map), 26, linear), n_epochs, value_map, learning_rate)
#     nnetwork.fit(x_train,y_train, x_test, y_test)
#     languages_to_classify = list(value_map.keys())
#     print('\n')
#     print("I can only classify those classes:", *languages_to_classify)
#     print('\n')
#     user_stopped = False
#
#     while not user_stopped:
#         user_input = input(f"Please give me text in either {languages_to_classify} languages:\nIn any other situation skip current "
#                            f"iteration.\nType"
#                            f"'stop' to end classification\n> ")
#
#         if user_input == 'stop':
#             print('End of classification')
#             break
#
#         try:
#             vector = vectorize(user_input)
#             predicted_as = nnetwork.predict(vector)
#             predicted_label = [key for key, val in value_map.items() if val == predicted_as]
#             print(f'\nPrediction of perceptron is: {predicted_as}')
#             print(f'Which means: {predicted_label[0]}\n')
#
#         except Exception as e:
#             print(e)
#             continue
#


if __name__ == '__main__':
    test()
    # #  For reference i use uniform weights, hence 6/(fanin+fanout)
    # parser = argparse.ArgumentParser(description="CNN with 1 layer from scratch")
    # parser.add_argument("-ds", "--dataset-path", type=str,
    #                     help="Provide a path to a dataset from current folder. Otherwise, give full path",
    #                     required=True)
    #
    # parser.add_argument("-e", type=int, help="Number of epochs",
    #                     required=False, default=30)
    # parser.add_argument("-a", "--alpha", type=float, help="Constant learning rate",
    #                     required=False, default=0.1)
    # parser.add_argument("-it",
    #                     "--iterate-mode",
    #                     type=int,
    #                     help="Iterative mode, you give me vectors I give you classes",
    #                     choices=[0, 1],
    #                     required=False,
    #                     default=0)
    #
    # args = parser.parse_args()
    #
    # if args.dataset_path is  None:
    #     raise Exception('You should provide path to datasets.')
    #
    # if args.iterate_mode == 1:
    #     iterate(args)
    #     exit(0)
    #
    # default(args)
