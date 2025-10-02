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
        out = Value(self.val + other.val, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
            
            self._backward() if self._backward else None
            other._backward() if other._backward else None
            
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float, Value)), "Only supports int/float/Value powers."
        other = other.val if isinstance(other, Value) else other
        out = Value(self.val ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.val ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        assert isinstance(other, (int, float, Value)), "Only supports int/float/Value multiplication."
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.val * other.val, (self, other), '*')

        def _backward():
            self.grad += other.val * out.grad
            other.grad += self.val * out.grad
        out._backward = _backward

        return out
    