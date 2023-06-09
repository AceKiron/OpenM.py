import numpy as np
import openm

def prime_numbers(n):
    primes = []
    for i in range(2, n + 1):
        for j in range(2, int(i ** 0.5) + 1):
            if i%j == 0:
                break
        else:
            primes.append(i)
    return primes

datasetInputs = []
datasetOutputs = prime_numbers(2000)
for i in range(len(datasetOutputs)):
    datasetInputs.append([i, i + 1])

model_filename = "prime_numbers2.pkl"

model = openm.Model(
    openm.Dataset(
        datasetInputs,
        datasetOutputs
    ),
    filename=model_filename
)

try:
    model.train(
        verbose=True,
        batchSize=50,
        saveEvery=500
    )
except KeyboardInterrupt:
    pass

model.save(True)