from .Nodes import *
from copy import deepcopy
import numpy as np
import pickle
import os

class Model:
    def __init__(self, dataset, filename=None):
        self.dataset = dataset
        self.filename = filename

        self.answersLength = len(dataset.data[1])
        self.inputsLength = len(dataset.data[0][0])

        if filename == None or not os.path.isfile(filename):
            self.rootNode = Node.bigMutate(self.inputsLength, 1e+10)
            self.inaccuracy = self.calculateInaccuracy(
                self.dataset.data[1],
                self.rootNode.visit(
                    self.dataset.data[0]
                )
            )
            self.mutationPower = np.exp(1 / self.inaccuracy)
        else:
            self.rootNode, self.mutationPower = pickle.load(open(filename, "rb"))
            self.inaccuracy = self.calculateInaccuracy(
                self.dataset.data[1],
                self.rootNode.visit(
                    self.dataset.data[0]
                )
            )
    
    def save(self, verbose):
        if self.filename != None:
            pickle.dump((self.rootNode, self.mutationPower), open(self.filename, "wb"))

            if verbose:
                print(
                    self.dataset.data[1] - self.rootNode.visit(
                        self.dataset.data[0]
                    )
                )

    def train(self, verbose=False, batchSize=50, saveEvery=500):
        gen = 0
        evolvedSinceLastSave = False

        while True:
            if self.inaccuracy == 0:
                break
            
            batch = {
                self.inaccuracy: self.rootNode
            }

            for i in range(batchSize):
                newNode = deepcopy(self.rootNode)

                mutationPowerMultiplier = 1 - 0.0001 * (i + np.random.random())

                if i % 2 == 0:
                    newNode = Node.bigMutate(self.inputsLength)
                else:
                    newNode.mutate(self.inputsLength, self.mutationPower * mutationPowerMultiplier)

                inaccuracy = self.calculateInaccuracy(
                    self.dataset.data[1],
                    newNode.visit(
                        self.dataset.data[0]
                    )
                )

                batch[inaccuracy] = newNode

            bestOutOfBatch = min(batch)
            gen += 1

            if bestOutOfBatch < self.inaccuracy:
                newNode = batch[bestOutOfBatch]

                if verbose:
                    print(self.inaccuracy, bestOutOfBatch, self.mutationPower, gen)
                    print(newNode)

                self.mutationPower *= .998

                self.rootNode = newNode
                self.inaccuracy = bestOutOfBatch

                evolvedSinceLastSave = True
            else:
                self.mutationPower *= .999
            
            if gen % saveEvery == saveEvery - 1 and evolvedSinceLastSave:
                self.save(verbose)
                evolvedSinceLastSave = False

    def calculateInaccuracy(self, expected, actual):
        return np.average(
            np.abs(
                expected - actual
            )
        )