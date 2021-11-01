import numpy as np
from activation import Activation


class Output(Activation):
    def __init__(self, activation, activation_diff, loss, loss_diff):
        super().__init__(activation, activation_diff)
        self._loss = loss
        self._loss_diff = loss_diff

    def calculate_loss(self, y_true, y_pred):
        return self._loss(y_true, y_pred)

    def calculate_loss_diff(self, y_true, y_pred):
        return self._loss_diff (y_true, y_pred)
