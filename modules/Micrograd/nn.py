import random
from __future__ import annotations
from value import Value

class Module:

    def zero_grad(self):
        """
        Description:
            Sets the gradients of alle parameters to zero
        """
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """
        Description:
            Returns the parameters
        """
        return []

class Neuron(Module):

    def __init__(self, nin: int, nonlin: bool = False) -> Neuron:
        """
        Description:
            Creates a neuron with nin weights.
        Inputs:
            nin: How many paramters the neuron should have
            nonlin: Whether a non-linear activation should be applied
        """
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, xs: list) -> Value:
        """
        Description:
            Given a list of xs the forward pass is executed
        Inputs:
            xs: A list of x-values
        Outputs:
            The calculated forward pass value
        """
        act = sum((wi*xi for wi,xi in zip(self.w, xs)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        """
        Description:
            Returns a list of parameters as well as the bias
        Outputs:
            The neuron parameters and bias
        """
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs) -> Layer:
        """
        Description:
            Creates a layer (list of neurons).
        Inputs:
            nin: Neuron dimensionality
            nout: Number of neurons in layer
        """
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, xs) -> Value:
        """
        Description:
            Calculates the output for each neuron with 
            the given input values
        Inputs:    
            xs: Input values
        Outputs:
            The calculated forward pass value
        """
        out = [n(xs) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> list:
        """
        Description:
            Returns a list of all parameters in the layer
        Ouputs:
            List of parameters
        """
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin: int, nouts: list) -> MLP:
        """
        Description:
            The multi-layer perceptron feeds the outputs of one 
            layer into the next. Here a list of layers is added 
            to the MLP object.
        Inputs:
            nin: Number of inputs to MLP
            nouts: List of outputs for each layer            
        """
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, xs: list) -> list | Value:
        """
        Description:
            Give inputs it calculates the MLP output via 
            a forward pass.
        Inputs:
            xs: Input values
        Outputs:
            The number af values given by the last layer
        """
        for layer in self.layers:
            xs = layer(xs)
        return xs

    def parameters(self):
        """
        Description:
            Returns a list of all parameters in the layer
        Ouputs:
            List of parameters
        """
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"