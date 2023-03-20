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
datasetOutputs = prime_numbers(1000)
for i in range(1, len(datasetOutputs) + 1):
    datasetInputs.append([i])

model_filename = "prime_numbers.pkl"

model = openm.Model(
    openm.Dataset(
        datasetInputs,
        datasetOutputs
    ),
    filename=model_filename
)

model.train(
    verbose=True,
    batchSize=50,
    saveEvery=500
)