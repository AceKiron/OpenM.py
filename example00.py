import numpy as np
import openm

# y = -2 * x1
model = openm.Model(
    openm.Dataset(
        [[-10], [-9], [-8], [-7], [-6], [-5], [-4], [-3], [-2], [-1], [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
        [20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10, -12, -14, -16, -18, -20]
    )
)
model.train(
    verbose=True,
    batchSize=10
)