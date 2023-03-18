import numpy as np

class Node:
    def __init__(self):
        raise NotImplementedError()
    
    def __str__(self):
        raise NotImplementedError()
    
    def bigMutate(inputsLength, i=0):
        if i < 5:
            x = np.int64(np.floor(np.random.random() * 5))
        else:
            x = np.int64(np.floor(np.random.random() * 2))
        
        if x == 0:
            return StaticNumberNode(np.random.random() - 0.5)
        elif x == 1:
            return VariableNode(np.int64(np.floor(np.random.random() * inputsLength)))
        elif x == 2:
            return SumNode(Node.bigMutate(inputsLength, i + 1), Node.bigMutate(inputsLength, i + 1))
        elif x == 3:
            return ProductNode(Node.bigMutate(inputsLength, i + 1), Node.bigMutate(inputsLength, i + 1))
        elif x == 4:
            return NaturalLogarithmNode(Node.bigMutate(inputsLength, i + 1))

    def mutate(self, inputsLength, mutationPower):
        raise NotImplementedError()

    def visit(self, inputs):
        raise NotImplementedError()

class StaticNumberNode(Node):
    def __init__(self, value=None):
        if value == None:
            value = np.random.random() - 0.5
        self.value = value

    def __str__(self):
        return "StaticNumber<" + str(self.value) + ">"

    def mutate(self, inputsLength, mutationPower):
        self.value += (np.random.random() - 0.5) * mutationPower

    def visit(self, inputs):
        return np.full((len(inputs)), self.value, dtype=np.float64)

class VariableNode(Node):
    def __init__(self, variableIndex=0):
        self.variableIndex = variableIndex
    
    def __str__(self):
        return "Variable<" + str(self.variableIndex) + ">"

    def mutate(self, inputsLength, mutationPower):
        self.variableIndex = np.int64(np.floor(np.random.random() * inputsLength))

    def visit(self, inputs):
        a = []
        for inp in inputs:
            a.append(np.float64(inp[self.variableIndex]))
        return a

class SumNode(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return "Sum<" + str(self.left) + ";" + str(self.right) + ">"

    def mutate(self, inputsLength, mutationPower):
        self.left.mutate(inputsLength, mutationPower)
        self.right.mutate(inputsLength, mutationPower)

    def visit(self, inputs):
        a = []
        for i in range(len(inputs)):
            a.append(self.left.visit(inputs)[i] + self.right.visit(inputs)[i])
        return a

class ProductNode(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
    def __str__(self):
        return "Product<" + str(self.left) + ";" + str(self.right) + ">"
    
    def mutate(self, inputsLength, mutationPower):
        self.left.mutate(inputsLength, mutationPower)
        self.right.mutate(inputsLength, mutationPower)

    def visit(self, inputs):
        a = []
        for i in range(len(inputs)):
            a.append(self.left.visit(inputs)[i] * self.right.visit(inputs)[i])
        return a

class NaturalLogarithmNode(Node):
    def __init__(self, child):
        self.child = child

    def __str__(self):
        return "NaturalLogarithm<" + str(self.child) + ">"
    
    def mutate(self, inputsLength, mutationPower):
        self.child.mutate(inputsLength, mutationPower)

    def visit(self, inputs):
        return np.log(self.child.visit(inputs))