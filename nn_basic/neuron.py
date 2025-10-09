from abc import ABC, abstractmethod

from value import Value
from math import tanh

class ParamModule(ABC):
    @abstractmethod
    def parameters(self) -> list:
        pass
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
    
class Neuron(ParamModule):
    def __init__(self, dimension, activation=tanh):
        import random
        self.w = [Value(random.uniform(-1, 1)) for _ in range (dimension)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation

    def set_activation(self, activation):
        if callable(activation):
            self.f = activation
        else:
            act_name = activation.lower()
            if act_name == 'tanh':
                self.f = lambda x: x.tanh()
            elif act_name == 'relu':
                self.f = lambda x: x.relu()
            elif act_name == 'sigmoid':
                self.f = lambda x: (1 / (1 + (-x).exp()))
            elif act_name == 'identity' or act_name == 'linear':
                self.f = lambda x: x
            else:
                raise ValueError(f"Unsupported activation: {activation}")

    def __call__(self, x):
        act = sum(((wi * xi) for wi, xi in zip(self.w, x)), self.b)
        return self.activation(act)
    
    def parameters(self):
        return self.w + [self.b]    
    
        
class Layer(ParamModule):
    def __init__(self, d_in, d_out):
        self.neurons = [Neuron(dimension=d_in) for _ in range(d_out)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons] if len(self.neurons) > 1 else self.neurons[0](x)
        return outs
    
    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]

class MLP(ParamModule):
    def __init__(self, d_in, d_outs):
        sz = [d_in] + d_outs
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(sz) - 1)]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
    

x = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0]]
y = [1.0, -1.0, -1.0]
mlp = MLP(3, [4, 8, 4, 1])


y_pred = [mlp(xi) for xi in x]



for epoch in range(100):
    mlp.zero_grad()
    y_pred = [mlp(xi) for xi in x]
    loss = sum(( (yi_p - yi)**2 for yi, yi_p in zip(y, y_pred) ))
    loss.grad = 1.0
    loss.backward()
    
    learning_rate = 0.01 * (0.97 ** epoch) if epoch < 10 else 0.005 * (0.97 ** (epoch - 10))
    
    for p in mlp.parameters():
        p.val += - learning_rate * p.grad
    
    print(f"Epoch {epoch}: loss {loss.val}")


