import numpy as np
import openm

# y = 2 * x1 + 1
model = openm.Model(
    openm.Dataset(
        [[-10], [-9], [-8], [-7], [-6], [-5], [-4], [-3], [-2], [-1], [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
        [-19, -17, -15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    )
)
model.train(
    verbose=True,
    batchSize=25
)