import argparse
import random
import sys
from typing import Optional

import matplotlib.pyplot as plt


# File read. test/train split
def read_file(url):
    with open(url, 'r') as f:
        file = f.readlines()
    return [line.strip("\n").split(",") for line in file]


def train_test_split(train_path, test_path):
    training_lines = read_file(train_path)
    test_lines = read_file(test_path)

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for line in training_lines:
        x_train.append([float(el) for el in line[:4]])
        y_train.append(line[-1])

    for line in test_lines:
        x_test.append([float(el) for el in line[:4]])
        y_test.append(line[-1])

    return x_train, x_test, y_train, y_test


# utils
def minus_of_list(x1: list, x2: list) -> list:
    return [x - y for x, y in zip(x1, x2)]


def sum_of_list(x1: list, x2: list) -> list:
    return [x + y for x, y in zip(x1, x2)]


def power_of_list(x: list, power):
    return [el**power for el in x]


def calculate_net_value(x1: list[float], w1: list[float]):
    return sum([x*w for x, w in zip(x1, w1)])


class Layer:
    __slots__ = (
        'neurons',
        'epochs',
        'x_train',
        'y_train',
        'n_workers',
        'alpha',
        'values',
        'is_trained'
    )

    def __init__(self, number_neurons=3, epochs=10, alpha=0.1):
        self.values: list = None
        self.n_workers = number_neurons
        self.epochs = epochs
        self.alpha = alpha
        self.x_train = None
        self.y_train = None
        self.neurons: Optional[list[Neuron]] = None
        self.is_trained = False

    def predict(self, x_t):
        assert self.is_trained != False, 'Neurons are not trained'
        y_preds = []
        for row in x_t:
            activations = [1 if neuron.is_activated(row) else 0 for neuron in self.neurons]
            predicted_class_index = activations.index(max(activations))
            y_preds.append(predicted_class_index)
        return y_preds

    def train(self):
        for epoch in range(self.epochs):
            # print(f"Starting epoch number: {epoch+1}")
            self.study_on_dataset()
            # print(f"Finished epoch number: {epoch+1}")

        self.is_trained = True

    def study_on_dataset(self):
        for (x_t, y_t) in zip(self.x_train, self.y_train):
            for neuron in self.neurons:
                actual_decision = 1 if neuron.is_activated(x_t) else 0
                correct_decision = 1 if self.values.index(y_t) == neuron.class_index else 0
                self.update_neuron(actual_decision, correct_decision, x_t, neuron)

        # currently only for first neuron will work.
            # neuron = self.neurons[0]
            # actual_decision = 1 if neuron.is_activated(x_t) else 0
            # correct_decision = self.values.get(y_t)
            # self.update_neuron(actual_decision, correct_decision, x_t, neuron)

    def update_neuron(self, actual_decision, correct_decision, x_t, neuron):
        if correct_decision == actual_decision:
            return

        # Right side of the equation
        coefficient = (correct_decision-actual_decision) * self.alpha
        calculated_vector = [coefficient*number for number in x_t]
        calculated_vector.append(coefficient*neuron.theta)

        # Left side
        old_weights = neuron.weights
        old_weights.append(neuron.theta)

        # Finale
        new_weights = sum_of_list(old_weights, calculated_vector)

        # Updating
        neuron.weights, neuron.theta = new_weights[:-1], new_weights[-1]

    def create_neurons(self, n_workers):
        # self.neurons = [Neuron(len(self.x_train[0])) for _ in range(n_workers)]
        self.neurons = [Neuron(len(self.x_train[0]), class_index) for class_index in range(n_workers)]

    def fit(self, x_train, y_train):
        if any(len(vec) != len(x_train[0]) for vec in x_train):
            raise ValueError("Inconsistent dimensions in the training data")
        self.x_train = x_train
        self.y_train = y_train
        self.create_neurons(self.n_workers)
        self.map_values()

    def map_values(self):
        # self.values = {class_index: index for index, class_index in enumerate(set(self.y_train))}
        self.values = [class_name for class_name in set(self.y_train)]
        # index = 0
        # self.values = {}
        # for el in set(self.y_train):
        #     self.values[el] = index
        #     index += 1


def get_random_weights(number_of_weights):
    return [random.choice([0, 1]) for _ in range(number_of_weights)]


class Neuron:
    __slots__ = (
        'weights',
        'theta',
        'class_index'
    )

    def __init__(self, number_of_arguments, class_index):
        self.weights = get_random_weights(number_of_arguments)
        self.theta = 0.1  # by default
        self.class_index = class_index

    def update_weights(self, new_weights):
        assert self.weights != new_weights, 'same weight'
        self.weights = new_weights

    def update_theta(self, new_theta):
        assert self.theta != new_theta, 'same theta'
        self.theta = new_theta

    def is_activated(self, vector: list[float]):
        net_value = calculate_net_value(vector, self.weights)
        return net_value >= self.theta


def accuracy_score_for_perceptron(y_preds, y_test, mapped_value):
    mapped_y_test = map(lambda x: mapped_value.index(x), y_test)
    accuracy_list = [1 if y_t == y_y else 0 for y_t, y_y in zip(mapped_y_test, y_preds)]
    accuracy_percent = accuracy_list.count(1) / len(accuracy_list)
    return accuracy_percent * 100


def accuracy_score(y_test, y_preds):
    assert len(y_test) == len(y_preds), 'Different shapes, error'

    accuracy_list = [1 if y_t == y_y else 0 for y_t, y_y in zip(y_test, y_preds)]
    accuracy_percent = accuracy_list.count(1) / len(accuracy_list)
    return accuracy_percent * 100

#
def plot_all_of_them():
    x_train, x_test, y_train, y_test = train_test_split("train.csv", "test.csv")

    plot_data = []
    for i in range(1, 1001):
        print(i)
        layer = Layer(epochs=i)
        layer.fit(x_train, y_train)
        layer.train()
        y_preds = layer.predict(x_test)
        mapped_values = layer.values
        plot_data.append(accuracy_score_for_perceptron(y_preds, y_test, mapped_values))

    plt.plot(plot_data)
    plt.ylabel("Accuracy")
    plt.xlabel("Value of \"k\"")
    plt.show()
#
#
# def main(args) -> None:
#     x_train, x_test, y_train, y_test = train_test_split(args.train_path, args.test_path)
#     knn = KNN(args.k)
#     knn.fit(x_train, y_train)
#     y_preds = knn.predict(x_test)
#     print(f"Accuracy: {accuracy_score(y_test, y_preds):.2f}%")
#
#
# def main_solo(args) -> None:
#     knn = KNN(args.k)
#     knn.fit(x_train, y_train)
#     my_list_vector = []
#     while len(my_list_vector) != len(x_train[0]):
#         str_inp = input('podaj wartosc> ')
#         my_list_vector.append(float(str_inp))
#
#     y_preds = knn.solo_predict(my_list_vector)
#     print(f"Predicted: {y_preds}")


if __name__ == '__main__':

    x_train, x_test, y_train, y_test = train_test_split("train.csv", "test.csv")
    layer = Layer(epochs=10000)
    layer.fit(x_train, y_train)
    layer.train()
    y_preds = layer.predict(x_test)
    mapped_values = layer.values
    print(mapped_values)
    print(y_preds)
    print(accuracy_score_for_perceptron(y_preds, y_test, mapped_values))

    # parser = argparse.ArgumentParser(description="KNN from scratch")
    # parser.add_argument("-trp","--train-path", type=str, help="Provide a path to a train dataset from current folder. Otherwise, give full path", required=False)
    # parser.add_argument("-tp","--test-path", type=str, help="Provide a path to a test dataset from current folder. Otherwise, give full path", required=False)
    # parser.add_argument("-k", type=int, help="Number of nearest neighbours", required=False, default=3)
    # parser.add_argument("-p", '--plot', type=int, help="Plot from k=1 to k=105. provide p=1 to use this option. Provide p=2 for solo vector", required=False, default=0)
    #
    # args = parser.parse_args()
    #
    # if args.train_path is None or args.test_path is None:
    #     raise Exception('You should provide path to datasets.')
    #
    # if args.plot == 1:
    #     plot_all_of_them(args)
    #     sys.exit()
    #
    # if args.plot == 2:
    #     main_solo(args)
    #     sys.exit()
    #
    # main(args)
    # # traning = 'train.csv'
    # # test = 'test.csv'