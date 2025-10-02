from value import Value
from math import tanh

class Neuron:
    def __init__(self, dimension):
        import random
        self.w = [Value(random.uniform(-1, 1)) for _ in range (dimension)]
        self.b = Value(random.uniform(-1, 1))
        self.f = lambda x: tanh(x)
    
    def activation(self, f):
        self.f = f
    
    def __call__(self, x):
        act = sum(((wi * xi) for wi, xi in zip(self.w, x)), self.b)
        return self.f(act.val)
        
class Layer:
    def __init__(self, d_in, d_out):
        self.neurons = [Neuron(dimension=d_in) for _ in range(d_out)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs

class MLP:
    def __init__(self, d_in, d_outs):
        sz = [d_in] + d_outs
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(sz) - 1)]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

m = MLP(3, [6, 6, 3])
print(m([1.0, 2.0, 3.0]))