import math

class Value:
    def __init__(self, val, label='unknown', _children=(), _op=None):
        self.val = val
        self.grad = 0.0
        self.label = label
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        
    def __repr__(self):
        return f"Value(val={self.val}, prev={self._prev}, op={self._op}, grad={self.grad})"
    
    def __add__(self, other):
        assert isinstance(other, (int, float, Value)), "Only supports int/float/Value addition."
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.val + other.val, _children=(self, other), _op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
            
        out._backward = _backward

        return out
    
    def __radd__(self, other):        
        return self + other
    
    def __sub__(self, other):
        other  = other if isinstance(other, Value) else Value(other)
        out = Value(self.val - other.val, _children=(self, other), _op='-')
        def _backward():
            self.grad += out.grad
            other.grad +=  out.grad * -1
            
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * -1
    
    def __rsub__(self, other):
        return -(self - other)
    
    def __pow__(self, other):
        assert isinstance(other, (int, float, Value)), "Only supports int/float/Value powers."
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.val ** other.val, _children=(self,), _op=f'**{other}')

        def _backward():
            self.grad += other.val * (self.val ** (other.val - 1)) * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        assert isinstance(other, (int, float, Value)), "Only supports int/float/Value multiplication."
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.val * other.val,_children= (self, other),_op= '*')

        def _backward():
            self.grad += other.val * out.grad
            other.grad += self.val * out.grad
            
        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other * (self ** -1)
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
    
    def tanh(self):
        x = self.val
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, _children=(self,), _op='tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out