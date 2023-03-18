import numpy as np
import openm

# y = ln(x1 * x2)
datasetInputs = []
datasetOutputs = []
for i in range(10):
    x = 1 / np.random.random()
    y = 1 / np.random.random()
    datasetInputs.append([x, y])
    datasetOutputs.append(np.log(x * y))

model = openm.Model(
    openm.Dataset(
        datasetInputs,
        datasetOutputs
    )
)
model.train(
    verbose=True,
    batchSize=10
)