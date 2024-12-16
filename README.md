# NeuroGrad
A simple scalar-valued Autograd Engine along with a simple Neural Net Library built on top of it (modelled after the PyTorch API)

The Autograd Engine implements backpropagation in a simple Neural Network Model (non-convolutional). The DAG operates on scalar data, and does not use vectorization. I shall upload a Demo Notebook showing the implementation a soon [hopefully :)]

### NOTE-
1. The Model supports only 1 neuron output layer architecture for now! (shall be updated soon)
2. The Model can implement only the following activations as of now-
    * Linear
    * Rectified Linear Unit (ReLU)
    * Hyperbolic tangent (tanh)
    * Sigmoid (sigmoid)
    * Exponential (exp)
3. The file [autograd/draw.py](./autograd/draw.py) implements Digraph from graphviz for Visualization and is not necessary for implementing a Neural Network
