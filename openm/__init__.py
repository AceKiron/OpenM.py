from .Model import Model

import numpy as np

class Dataset:
    def __init__(self, inputs, outputs):
        self.data = (
            np.float64(inputs),
            np.float64(outputs)
        )