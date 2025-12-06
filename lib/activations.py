# el file da fe kol el activation functions el basta5dmha fl neural network (ReLU, Sigmoid, Tanh, Softmax)
# 3ashan a7awwel el output ben el layers w a7ot non-linearity fl model

import numpy as np
from .layers import Layer

class ReLU(Layer):
    """Rectified Linear Unit activation"""

    def __init__(self):
        self.X = None
    
    def forward(self, X):
        """
        ReLU forward: f(x) = max(0, x)
    
        Args:
            X: input array
        
        Returns:
            Y: ReLU output
        """
        self.X = X      # hena ba-store el input 3ashan a7tago fl backward
        Y = np.maximum(0, X)   # hena ba-apply relu 3ala kol element
        return Y
    
    def backward(self, dL_dY):
        """
        ReLU backward: derivative is 1 if X > 0, else 0
        
        Args:
            dL_dY: gradient from next layer
        
        Returns:
            dL_dX: gradient for previous layer
        """
        mask = (self.X > 0).astype(float)   # hena ba3mel mask 3ashan el negative values teb2a zero fl gradient
        dL_dX = dL_dY * mask               # el gradient by3ady bas ll values el kanet > 0
        return dL_dX

class Sigmoid(Layer):
    """Sigmoid activation"""

    def __init__(self):
        self.Y = None
    
    def forward(self, X):
        """
        Sigmoid forward: f(x) = 1 / (1 + e^(-x))
        
        Args:
            X: input array
        
        Returns:
            Y: sigmoid output
        """
        X_clipped = np.clip(X, -500, 500)  # hena ba-limit el values 3ashan ma7salsh overflow
        self.Y = 1 / (1 + np.exp(-X_clipped))  # sigmoid function el ma3rofa
        return self.Y
    
    def backward(self, dL_dY):
        """
        Sigmoid backward: derivative is sigmoid(x) * (1 - sigmoid(x))
        
        Args:
            dL_dY: gradient from next layer
        
        Returns:
            dL_dX: gradient for previous layer
        """
        dL_dX = dL_dY * self.Y * (1 - self.Y)  # derivative beta3 sigmoid mo3tamed 3ala el output nafsaha
        return dL_dX

class Tanh(Layer):
    """Tanh activation"""

    def __init__(self):
        self.Y = None
    
    def forward(self, X):
        """
        Tanh forward: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        
        Args:
            X: input array
        
        Returns:
            Y: tanh output
        """
        self.Y = np.tanh(X)    # tanh direct mn numpy
        return self.Y
    
    def backward(self, dL_dY):
        """
        Tanh backward: derivative is 1 - tanh(x)^2
        
        Args:
            dL_dY: gradient from next layer
        
        Returns:
            dL_dX: gradient for previous layer
        """
        dL_dX = dL_dY * (1 - self.Y ** 2) 
        return dL_dX

class Softmax(Layer):
    """Softmax activation"""

    def __init__(self):
        self.Y = None
    
    def forward(self, X):
        """
        Softmax forward: f(x) = e^x / sum(e^x)
        
        Args:
            X: input array, shape (batch_size, num_classes)
        
        Returns:
            Y: softmax output
        """
        X_shifted = X - np.max(X, axis=1, keepdims=True)  # hena ba-shift el inputs 3ashan stability fl exponentials
        exp_X = np.exp(X_shifted)
        self.Y = exp_X / np.sum(exp_X, axis=1, keepdims=True)  # normalizing 3ashan el probabilities t3mel 1
        return self.Y
    
    def backward(self, dL_dY):
        """
        Softmax backward
        
        Args:
            dL_dY: gradient from next layer
        
        Returns:
            dL_dX: gradient for previous layer
        """
        batch_size = self.Y.shape[0]        # hena kont 3ayez el batch size bas mn el dimension el awel
        dL_dX = np.zeros_like(self.Y)
        
        for i in range(batch_size):
            jacobian = np.diag(self.Y[i]) - np.outer(self.Y[i], self.Y[i])
            dL_dX[i] = np.dot(jacobian, dL_dY[i])   # hena ba-7seb gradient ll input el asli mn gradient el next layer
        
        return dL_dX
