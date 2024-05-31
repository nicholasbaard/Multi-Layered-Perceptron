import numpy as np

class Sigmoid:
    # Sigmoid activation function
    def __call__(self, x, derivative=False):
        if derivative:
            # Derivative of sigmoid function
            return self.__call__(x)*(1 - self.__call__(x))
        # Sigmoid function
        return 1/(1 + np.exp(-x))
    
class Softmaxzero:
    # Softmax activation function
    def __call__(self, x):
        # Softmax function
        return np.exp(x)/np.sum(np.exp(x), axis=0, keepdims=True)
    
class Softmaxone:
    # Softmax activation function
    def __call__(self, x):
        # Softmax function
        return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)
class Tanh:
    # Hyperbolic tangent activation function
    def __call__(self, x, derivative=False):
        if derivative:
            # Derivative of hyperbolic tangent function
            return 1/(np.square(x) + 1)
        # Hyperbolic tangent function
        return 2/(1 + np.exp(-2*x)) - 1

class Relu:
    # Rectified Linear Unit activation function
    def __call__(self, x, derivative=False):
        if derivative:
            # Derivative of ReLU function
            return 1 if(x >= 0) else 0
        # ReLU function
        return np.maximum(0, x)
