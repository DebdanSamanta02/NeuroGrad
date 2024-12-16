import math, random
import numpy as np
import matplotlib.pyplot as plt

class Value:
    ''' STORES A SCALAR and IT'S GRADIENT'''

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda : None

        #variables for graph
        self._prev = set(_children)
        self._op = _op
        self.label = label

    
    '''
        MATHEMATICAL EXPRESSIONS
    '''
    def __add__(self, other):
        #self + other

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out
    def __radd__(self, other):
        #other + self

        return self + other

    def __mul__(self, other):
        #self * other

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    def __rmul__(self, other):
        #other * self

        return self * other
    
    def __pow__(self, other):
        #self ** other

        assert isinstance(other, (int, float)), "only int/float exponents"
        out = Value(self.data ** other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        #self / other

        return self * (other ** -1)
    def __rtruediv__(self, other):
        #other / self

        return other * (self ** -1)
    
    def __neg__(self):
        #-self

        return self * -1

    def __sub__(self, other):
        #self - other

        return self + (-other)
    def __rsub__(self, other):
        #other - self

        return other + (-self)

    
    '''
        MATHEMATICAL FUNCTIONS
    '''
    def exp(self):
        #e ^ x

        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out
        
    def relu(self):
        #relu(x) = [x if x > 0 else 0]

        out = Value(max(self.data, 0), (self, ), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        
        return out

    def tanh(self):
        #tanh(x) = [e^x - 1]/[e^x - 1] -> output between -1.0 and 1.0

        x = self.data
        t = math.exp(2 * x)
        expr = (t - 1) / (t + 1)
        out = Value(expr, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - expr ** 2) * out.grad
        out._backward = _backward
        
        return out

    def sigmoid(self):
        #sigmoid(x) = [1 / (1 + e^x)] -> output between 0 and 1

        x = self.exp()
        return x / (1 + x)

    #representation
    def __repr__(self):
        return f"Value(data={self.data})"

    #back propagation loop
    def backward(self):
        #topological sort of children
        topo = []
        vis = set()
        def toposort(node):
            if node not in vis:
                vis.add(node)
                for child in node._prev:
                    toposort(child)
                topo.append(node)
        toposort(self)
        topo = topo[::-1]

        self.grad = 1.0
        for node in topo:
            #evaluates gradient in chain rule
            node._backward()