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


class ActivationFunction(ABC):
    @abstractmethod
    def activate(self, x):
        """
        Applies activation function on net value

        Parameters:
        - x: Input tensor or array.

        Returns:
        - Output number
        """
        pass

    @abstractmethod
    def derivative(self, x):
        """
        Uses derivative version of an activation function on net value

        Parameters:
        - x: Input tensor or array.

        Returns:
        - Output number
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
