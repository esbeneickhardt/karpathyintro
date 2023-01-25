from __future__ import annotations
import math

class Value:
    def __init__(self, data:float, _children:tuple = (), _op:str = '', label=''):
        """
        Description:
            A value object keeps track of how the value was calculated
            as well as how to calculate the value's gradient.
        Inputs:
            data: The value of the object
            _children: The values used to calculate the self.data
            _op: The operator to calculate self.data
            label: The name of the value object
        """
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __add__(self, other:Value) -> Value:
        """
        Description:
            Adds the current value to an input value
            and adds the calculation to the expression tree 
            plus adds how to calculate the gradient.\
        Inputs:
            other: A value
        Outputs:
            The sum of self and other
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out
    
    def __mul__(self, other:Value) -> Value:
        """
        Description:
            Multiplies the current value with an input value
            and adds the calculation to the expression tree 
            plus adds how to calculate the gradient.
        Inputs:
            other: A value
        Outputs:
            The product of self and other
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward            
        
        return out
    
    def __pow__(self, other:int | float) -> Value:
        """
        Description:
            Raises the current value to the power of an input value
            and adds the calculation to the expression tree 
            plus adds how to calculate the gradient.
        Inputs:
            other: A value
        Outputs:
            The product of self and other
        """
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out  
    
    def relu(self) -> Value:
        """
        Description:
            Rounds values lower that zero to zero
            and adds the calculation to the expression tree 
            plus adds how to calculate the gradient.
        Outputs:
            The zero or the value
        """
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out     

    def tanh(self) -> Value:
        """
        Description:
            Squeezes values between -1 and 1
            and adds the calculation to the expression tree 
            plus adds how to calculate the gradient.
        Outputs:
            tanh of the value
        """
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad = (1 - t**2) * out.grad
        out._backward = _backward

        return out 
    
    def backward(self) -> None:
        """
        Description:
            Iterates backwards through the expression tree from 
            the ouput to the inputs applying the chain rule, 
            thus calculating the individual gradients with 
            respect to the output.
        """
        # Sorts topologically from root to leafs
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        # Applying chain rule by going from root to leafs
        self.grad = 1
        for v in reversed(topo):
            v._backward()
    
    def __neg__(self) -> Value:
        """
        Description:
            Multiplies self with negative one.
        Outputs:
            The self value multiplied with negative one
        """
        return self * -1

    def __sub__(self, other:Value) -> Value:
        """
        Description:
            Subracts an input value from the current value
            and adds the calculation to the expression tree 
            plus adds how to calculate the gradient.
        Inputs:
            other: A value
        Outputs:
            The difference of self and other
        """
        return self + (-other)

    def __rmul__(self, other: Value) -> Value:
        """
        Description:
            Due to python's evaluation order it can calculate:
                Value * 3 <=> Value.__mul__(3)
            But not:
                3 * Value <=> 3.__mul__(Value)
            But by implementing the rmul special method a fallback
            is created that handles this by changing the evaluation
            order.
        """
        return self * other
    
    def __rsub__(self, other: Value) -> Value:
        return other + (-self)
    
    def __radd__(self, other: Value) -> Value:
        return self + other
    
    def __truediv__(self, other: Value) -> Value:
        """
        Description:
            Divides the current value with an input value
            and adds the calculation to the expression tree 
            plus adds how to calculate the gradient.
        Inputs:
            other: A value
        Outputs:
            self/other
        """
        return self * other**-1

    def __rtruediv__(self, other: Value) -> Value:
        """
        Description:
            Divides the input value with the current value
            and adds the calculation to the expression tree 
            plus adds how to calculate the gradient.
        Inputs:
            other: A value
        Outputs:
            other/self
        """
        return other * self**-1
    
    def __repr__(self):
        return f'Value(data={self.data}, grad={self.grad})'