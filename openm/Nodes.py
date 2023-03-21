import numpy as np

class Node:
    def __init__(self):
        raise NotImplementedError()
    
    def __str__(self):
        raise NotImplementedError()
    
    def bigMutate(inputsLength, i=0):
        if i < 30:
            x = np.int64(np.floor(np.random.random() * 9))
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
            return DivideNode(Node.bigMutate(inputsLength, i + 1), Node.bigMutate(inputsLength, i + 1))
        elif x == 5:
            return PowerNode(Node.bigMutate(inputsLength, i + 1), Node.bigMutate(inputsLength, i + 1))
        elif x == 6:
            return NaturalLogarithmNode(Node.bigMutate(inputsLength, i + 1))
        elif x == 7:
            return SineNode(Node.bigMutate(inputsLength, i + 1))
        elif x == 8:
            return CosineNode(Node.bigMutate(inputsLength, i + 1))

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
        return np.array(inputs, dtype=np.float64)[:,self.variableIndex]

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
        return self.left.visit(inputs) + self.right.visit(inputs)

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
        return self.left.visit(inputs) * self.right.visit(inputs)

class DivideNode(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
    def __str__(self):
        return "Division<" + str(self.left) + ";" + str(self.right) + ">"
    
    def mutate(self, inputsLength, mutationPower):
        self.left.mutate(inputsLength, mutationPower)
        self.right.mutate(inputsLength, mutationPower)

    def visit(self, inputs):
        return self.left.visit(inputs) / self.right.visit(inputs)

class PowerNode(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
    def __str__(self):
        return "Power<" + str(self.left) + ";" + str(self.right) + ">"
    
    def mutate(self, inputsLength, mutationPower):
        self.left.mutate(inputsLength, mutationPower)
        self.right.mutate(inputsLength, mutationPower)

    def visit(self, inputs):
        return self.left.visit(inputs) ** self.right.visit(inputs)

class NaturalLogarithmNode(Node):
    def __init__(self, child):
        self.child = child

    def __str__(self):
        return "NaturalLogarithm<" + str(self.child) + ">"
    
    def mutate(self, inputsLength, mutationPower):
        self.child.mutate(inputsLength, mutationPower)

    def visit(self, inputs):
        return np.log(self.child.visit(inputs))

class SineNode(Node):
    def __init__(self, child):
        self.child = child

    def __str__(self):
        return "Sine<" + str(self.child) + ">"
    
    def mutate(self, inputsLength, mutationPower):
        self.child.mutate(inputsLength, mutationPower)

    def visit(self, inputs):
        return np.sin(self.child.visit(inputs))

class CosineNode(Node):
    def __init__(self, child):
        self.child = child

    def __str__(self):
        return "Cosine<" + str(self.child) + ">"
    
    def mutate(self, inputsLength, mutationPower):
        self.child.mutate(inputsLength, mutationPower)

    def visit(self, inputs):
        return np.cos(self.child.visit(inputs))