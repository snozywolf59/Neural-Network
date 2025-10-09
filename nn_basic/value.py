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
        return f"Value(val={self.val}, grad={self.grad})"
    
    # --- Các toán tử cơ bản ---
    def __add__(self, other):
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
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.val - other.val, _children=(self, other), _op='-')
        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        out._backward = _backward
        return out
    
    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other - self
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.val * other.val, _children=(self, other), _op='*')

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
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Exponent must be int or float"
        out = Value(self.val ** other, _children=(self,), _op=f'**{other}')

        def _backward():
            self.grad += other * (self.val ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * -1
    
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
    
    # --- Các hàm kích hoạt ---
    def tanh(self):
        x = self.val
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, _children=(self,), _op='tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.val
        s = 1 / (1 + math.exp(-x))
        out = Value(s, _children=(self,), _op='sigmoid')
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        x = self.val
        out = Value(x if x > 0 else 0, _children=(self,), _op='relu')
        def _backward():
            self.grad += (1.0 if x > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def leaky_relu(self, alpha=0.01):
        x = self.val
        out = Value(x if x > 0 else alpha * x, _children=(self,), _op='leaky_relu')
        def _backward():
            self.grad += (1.0 if x > 0 else alpha) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.val
        out = Value(math.exp(x), _children=(self,), _op='exp')
        def _backward():
            self.grad += out.val * out.grad
        out._backward = _backward
        return out

    def log(self):
        x = self.val
        assert x > 0, "log undefined for non-positive values"
        out = Value(math.log(x), _children=(self,), _op='log')
        def _backward():
            self.grad += (1 / x) * out.grad
        out._backward = _backward
        return out

    def identity(self):
        # f(x) = x
        out = Value(self.val, _children=(self,), _op='identity')
        def _backward():
            self.grad += 1.0 * out.grad
        out._backward = _backward
        return out