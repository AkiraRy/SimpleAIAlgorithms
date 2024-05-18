from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Tuple, List


class Dataset(ABC):
    def __init__(self, dataset_path: Union[Path, str], test_set_percentage: float, batch_size: int = 1):
        self.dataset_path = dataset_path
        self.test_set_percentage = test_set_percentage
        self.batch_size = batch_size

    @abstractmethod
    def get_next_data_row(self) -> Tuple[List[str], 'np.ndarray']:
        pass

    @abstractmethod
    def get_next_test_row(self) -> Tuple[List[str], 'np.ndarray']:
        pass


class LibMath(ABC):
    @staticmethod
    @abstractmethod
    def apply_filter(first2darray: 'ndarray', filter: 'ndarray') -> Union[float, int]:
        pass


class Activation(ABC):
    @abstractmethod
    def forward(self, x):
        """
        Applies activation function on net value

        Parameters:
        - x: Input tensor or array.

        Returns:
        - Output tensor, array, number
        """
        pass

    @abstractmethod
    def backward(self, dO):
        """
        Backward pass method for the layer.

        Parameters:
            dO (numpy.ndarray): The gradient of the loss with respect to the output of the activation function of the current layer.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of the layer.
        """
        pass


class Layer(ABC):
    @abstractmethod
    def forward(self, Input):
        pass

    @abstractmethod
    def backward(self):
        pass


# class TorchMath(LibMath):
#
#     def add(self, a, b):
#         return torch.add(a, b)
#
#     def multiply(self, a, b):
#         return torch.mul(a, b)
