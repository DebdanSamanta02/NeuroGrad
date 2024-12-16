import math, random
import numpy as np
import matplotlib.pyplot as plt

from grad import Value

class Neuron:
    def __init__(self, n_inputs, activation = 'linear'):
        self.w = [Value(random.uniform(-1,1)) for _ in range(n_inputs)]
        self.b = Value(random.uniform(-1,1))
        self.activation = activation.lower()

    def __call__(self, x):
        # w dot x + b
        z = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        
        if self.activation == 'linear':
            return z
        elif self.activation == 'relu':
            return z.relu()
        elif self.activation == 'tanh':
            return z.tanh()
        elif self.activation == 'sigmoid':
            return z.sigmoid()

    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{self.activation} -> Neuron(weight: {len(self.w)}; bias: 1)"

class Layer:
    def __init__(self, n_inputs, n_outputs, activation = 'linear'):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_outputs)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(node) for node in self.neurons)}]"

class MLP:
    def __init__(self, n_inputs, n_outs, nonlinearity):
        #number of inputs, list of *neurons per layer, list of *corresponding activations
        
        temp = [n_inputs] + n_outs
        self.layers = [Layer(temp[i], temp[i+1], nonlinearity[i]) for i in range(len(temp) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

    def predict(self, input_set):
        y_predicted = [self(x)[0] for x in input_set]
        return y_predicted

    def squared_error_loss(self, input_set, output_set):
        y_predicted = self.predict(input_set)
        loss = sum((y_label - y_true) ** 2 for y_label, y_true in zip(y_predicted, output_set))
        return loss

    def fit(self, input_set, output_set, iters, alpha = 0.01):
        for i in range(iters):
            #forward prop
            loss = self.squared_error_loss(input_set, output_set)
        
            #back prop
            #zero-grad implementation
            for param in self.parameters():
                param.grad = 0.0
            loss.backward()
        
            #update
            for param in self.parameters():
                param.data -= alpha * param.grad
        
            #print
            if(i % 100 == 0):
                print(f"Iteration {i+1}: loss = {loss.data}")
            if(i == iters - 1):
                print(f"Iteration {i+1}: loss = {loss.data}")