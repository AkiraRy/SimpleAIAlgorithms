# returns a dictionary containing paths(value) to each letter (key)
import math
import os
from typing import Union, Tuple, List

from matplotlib import pyplot as plt

from AbstractClasses import Dataset
from PIL import Image
import numpy as np
from LayerImplementation import ConvolutionalLayer

def get_image_paths(folder_path):
    image_paths_dict = {}
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path) and item_path.lower().endswith('.png'):
            parent_folder = os.path.basename(folder_path)
            if parent_folder not in image_paths_dict:
                image_paths_dict[parent_folder] = []
            image_paths_dict[parent_folder].append(item_path)
        elif os.path.isdir(item_path):
            nested_paths = get_image_paths(item_path)
            for key, value in nested_paths.items():
                image_paths_dict.setdefault(key, []).extend(value)
    return image_paths_dict


def convert_paths_to_array(image_paths: list):
    converted_paths = []
    for img_path in image_paths:
        image_data = Image.open(img_path).convert("L")
        image_array = np.asarray(image_data) / 255
        converted_paths.append(image_array)
    return converted_paths


def get_numpy_arrays(dict_paths: dict[str, list[str]]):
    dict_numpy = dict()
    for key, value in dict_paths.items():
        new_value = convert_paths_to_array(value)
        dict_numpy.setdefault(key, []).extend(new_value)

    return dict_numpy

# also another problem should i shuffle all those value lists for each key or shuffle paths instead of numpy arrays
#  and should i shuffle list of keys?
# i mean so that we always go in different orders? but the length is still the same
# so do i shuffle list of keys every 7 iterations?
# would be better to have somewhere some strategy for sorting it

# i cant just shuffle, because min index or max index depends on the posotion of keys
# or i can just make a tuple of them like (key, min/max_index) and shuffle tuples in a list instead
# so that i can add some randomness


class DatasetImages(Dataset):
    __slots__ = (
        "numpy_dict",
        "test_set_percentage",
        "batch_size",
        "max_index_for_each_key_train",
        "list_of_keys",
        "max_len",
    )

    def __init__(self, dataset_path, test_set_percentage, batch_size=1):
        super().__init__(dataset_path, test_set_percentage, batch_size)
        self.numpy_dict = get_numpy_arrays(get_image_paths(self.dataset_path))
        self.list_of_keys = list(self.numpy_dict.keys())
        self.max_len = 0
        self.initialize_max_values()

    def get_next_test_row(self) -> Tuple[List[str], np.ndarray]:
        current_index = 0  # corresponds to the letter
        current_value_index = min(self.max_index_for_each_key_train)  # corresponds to the image value in an array

        while current_value_index < self.max_len:
            batch_keys = []
            batch_values = []

            for _ in range(self.batch_size):
                if current_index % len(self.list_of_keys) == 0 and current_index != 0:
                    current_index = 0
                    current_value_index += 1

                key = self.list_of_keys[current_index]
                values_for_key = self.numpy_dict.get(key)

                if current_value_index < self.max_index_for_each_key_train[current_index]:
                    current_index += 1
                    continue

                if not current_value_index < len(values_for_key):
                    current_index += 1
                    continue

                batch_keys.append(key)
                batch_values.append(values_for_key[current_value_index])
                current_index += 1

            if batch_keys:
                yield batch_keys, np.array(batch_values)

    def get_next_data_row(self) -> Tuple[List[str], np.ndarray]:
        current_index = 0  # corresponds to the letter
        current_value_index = 0  # corresponds to the image value
        max_value_index = max(self.max_index_for_each_key_train)

        while current_value_index < max_value_index:
            batch_keys = []
            batch_values = []

            for _ in range(self.batch_size):
                if current_index % len(self.list_of_keys) == 0 and current_index != 0:
                    current_index = 0
                    current_value_index += 1

                key = self.list_of_keys[current_index]
                values_for_key = self.numpy_dict.get(key)

                if current_value_index >= self.max_index_for_each_key_train[current_index]:
                    current_index += 1
                    continue

                if not current_value_index < len(values_for_key):
                    current_index += 1
                    continue
                batch_keys.append(key)
                batch_values.append(values_for_key[current_value_index])
                current_index += 1
            if batch_keys:
                yield batch_keys, np.array(batch_values)

    def initialize_max_values(self):
        self.max_index_for_each_key_train = []

        for _, value in self.numpy_dict.items():
            actual_value_len = len(value)
            if actual_value_len > self.max_len:
                self.max_len = actual_value_len

            allowed_len = 1 - self.test_set_percentage
            self.max_index_for_each_key_train.append(math.ceil(actual_value_len * allowed_len))

# from collections import Counter

if __name__ == '__main__':
    # Testing if this thing works correctly

    my_generator = DatasetImages("dataset", 0.2, batch_size=1)
    # dummy_list1 = {}
    dummy_list1 = []
    # dummy_list2 = {}
    dummy_list2 = []

    getter = my_generator.get_next_data_row()
    # for key, value in my_generator.get_next_data_row():
        # dummy_list1.setdefault(key, []).extend([value])
        # dummy_list1.append((key, value))
    dummy_list1.append(next(getter))

    its_key = dummy_list1[0][0]
    first_array = dummy_list1[0][1][0]
    print(f"{first_array.shape}")
    conv_layer = ConvolutionalLayer(kernel_size=(20,20), num_filters=4, stride=3, padding=0, use_bias=True)
    conv_layer2 = ConvolutionalLayer(kernel_size=(11,11), num_filters=4, stride=2, padding=0, use_bias=True)
    conv_layer3 = ConvolutionalLayer(kernel_size=(11,11), num_filters=4, stride=1, padding=0, use_bias=True)

    forward_pass_second = conv_layer3.forward(first_array)[0] # 0 is output,  1 is cache
    forward_pass_first = conv_layer2.forward(forward_pass_second[0])[0]
    forward_pass = conv_layer.forward(forward_pass_first[0])[0]
    new_array = forward_pass
    new_shape = forward_pass[0].shape

    print(f"{new_shape = }")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 5, 1)
    plt.imshow(first_array, cmap='gray')  # Use cmap='gray' for grayscale
    plt.title('Original Image')
    plt.axis('off')

    num_filters = new_array.shape[0]

    for i in range(num_filters):
        plt.subplot(1, num_filters + 1, i + 2)
        plt.imshow(forward_pass[i], cmap='gray')  # Use cmap='gray' for grayscale
        plt.title(f'Processed Image {i + 1}')
        plt.axis('off')

    # Display the new image
    # plt.subplot(1, 2, 2)
    # plt.imshow(new_array, cmap='gray')  # Use cmap='gray' for grayscale
    # plt.title('Processed Image')
    # plt.axis('off')

    plt.show()

    # old_im = Image.fromarray(first_array)
    # new_im = Image.fromarray(new_array)
    # old_im.show()
    # new_im.show()

    # Flatten the list of lists
    # flattened_list = [item for sublist in dummy_list1 for item in sublist]
    #
    # Count the occurrences of each letter
    # letter_occurrences = Counter(flattened_list)
    #
    # Print the dictionary
    # sorted_occurrences = dict(sorted(letter_occurrences.items()))
    #
    # Print the sorted dictionary
    # print(sorted_occurrences)

    # Populating dictionary from dummy_list2
    # for key, value in my_generator.get_next_test_row():
    #     # dummy_list2.setdefault(key, []).extend([value])
    #     dummy_list2.append(key)
    #
    # flattened_list2 = [item for sublist in dummy_list2 for item in sublist]
    #
    # # Count the occurrences of each letter
    # letter_occurrences2 = Counter(flattened_list2)
    #
    # # Print the dictionary
    # sorted_occurrences2 = dict(sorted(letter_occurrences2.items()))
    #
    # # Print the sorted dictionary
    # print(sorted_occurrences2)
    # #
    # sum_for_test = 0
    # for el in dummy_list2:
    #     sum_for_test+=len(el)
    #     print(f"length of the batches in data test: {el[1]}")

    # print(len(dummy_list1))
    # print(len(dummy_list2))

    # print(sum_for_train)
    # print(sum_for_test)
    # Find common elements
    # common_elements = set(dummy_list1) & set(dummy_list2)
    #
    # # Remove common elements from each array
    # array1 = [x for x in dummy_list1 if x not in common_elements]
    # array2 = [x for x in dummy_list2 if x not in common_elements]
    #
    # # Print arrays after removing common elements
    # print("Array 1 after removal:", len(array1))
    # print("Array 2 after removal:", len(array2))

    """
    1432
    350
    1782
    """
