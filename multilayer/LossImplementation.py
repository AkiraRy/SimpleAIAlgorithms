import numpy as np

from AbstractClasses import Loss


class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__("CrossEntropyLoss")

    def calculate_loss(self, y_true, y_predicted):
        return np.sum(-1 * np.sum(y_true * np.log(y_predicted), axis=0))

