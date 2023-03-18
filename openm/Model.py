from .Nodes import *
from copy import deepcopy
import numpy as np

class Model:
    def __init__(self, dataset):
        self.dataset = dataset

        self.answersLength = len(dataset.data[1])
        self.inputsLength = len(dataset.data[0][0])

        self.rootNode = Node.bigMutate(self.inputsLength, 1e+10)
    
    def train(self, verbose=False, batchSize=50):
        self.inaccuracy = self.calculateInaccuracy(
            self.dataset.data[1],
            self.rootNode.visit(
                self.dataset.data[0]
            ),
            self.answersLength
        )

        self.mutationPower = np.exp(1 / self.inaccuracy)

        gen = 0

        while True:
            if self.inaccuracy == 1:
                break
            
            batch = {
                self.inaccuracy: self.rootNode
            }

            for i in range(batchSize):
                newNode = deepcopy(self.rootNode)

                if i % 2 == 0:
                    newNode = Node.bigMutate(self.inputsLength)
                else:
                    newNode.mutate(self.inputsLength, self.mutationPower)

                inaccuracy = self.calculateInaccuracy(
                    self.dataset.data[1],
                    newNode.visit(
                        self.dataset.data[0]
                    ),
                    self.answersLength
                )

                batch[inaccuracy] = newNode

            bestOutOfBatch = min(batch)

            if verbose:
                print("Best of generation", gen, ":", bestOutOfBatch, batch[bestOutOfBatch])
            gen += 1

            if bestOutOfBatch < self.inaccuracy:
                newNode = batch[bestOutOfBatch]

                if verbose:
                    print(self.inaccuracy, bestOutOfBatch, self.mutationPower)
                    print(newNode)

                self.rootNode = newNode
                self.inaccuracy = bestOutOfBatch

                self.mutationPower /= np.exp(1 / bestOutOfBatch)

    def calculateInaccuracy(self, expected, actual, answersLength):
        return np.sum(
            np.exp(
                np.abs(
                    expected - actual
                )
            ) / answersLength
        )