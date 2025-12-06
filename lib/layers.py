# el file da 3ameleno 3ashan n7ot base class ll layers w Dense layer el hya el main building block fl neural network beta3na

import numpy as np

class Layer:
    """Base class for all layers"""

    
    def forward(self, X):
        """Forward pass - takes input and returns output"""
        raise NotImplementedError
    
    def backward(self, dL_dY):
        """Backward pass - takes gradient from next layer, returns gradient for previous layer"""
        raise NotImplementedError
    
    def get_params(self):
        """Return parameters (weights, biases) if any"""
        return []
    
    def get_grads(self):
        """Return gradients of parameters"""
        return []
    
class Dense(Layer):
    """Fully connected (Dense) layer"""


    def __init__(self, input_size, output_size):
        """
        Initialize Dense layer
        
        Args:
            input_size: number of input features
            output_size: number of output features
        """
        self.input_size = input_size    # hena m7afzeen size el input 3ashan n3raf shakl el matrix
        self.output_size = output_size   # w da size el output beta3 el layer
        
        # Initialize weights randomly
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        
        # Initialize biases to zero
        self.b = np.zeros((1, output_size))
        
        # Store for backward pass
        self.X = None
        self.dL_dW = None
        self.dL_db = None
    
    def forward(self, X):
        """
        Forward pass: Y = X @ W + b
        
        Args:
            X: input of shape (batch_size, input_size)
        
        Returns:
            Y: output of shape (batch_size, output_size)
        """
        self.X = X   # bn-store el input 3ashan n7tago fl backward
        Y = np.dot(X, self.W) + self.b   # el equation el asasyya ll dense layer
        return Y
    
    def backward(self, dL_dY):
        """
        Backward pass: calculate gradients
        
        Args:
            dL_dY: gradient of loss w.r.t. output, shape (batch_size, output_size)
        
        Returns:
            dL_dX: gradient of loss w.r.t. input, shape (batch_size, input_size)
        """
        batch_size = self.X.shape  # hena bas bngeb 3adad el samples lw 7tagnah
        
        # Gradient w.r.t. weights: dL/dW = X^T @ dL/dY
        self.dL_dW = np.dot(self.X.T, dL_dY)   # gradient 3ala el weights m3tamed 3ala input w gradient el output
        
        # Gradient w.r.t. bias: dL/db = sum(dL/dY) over batch
        self.dL_db = np.sum(dL_dY, axis=0, keepdims=True)   # el bias gradient howa sum el gradients 3ala el batch
        
        # Gradient w.r.t. input: dL/dX = dL/dY @ W^T
        dL_dX = np.dot(dL_dY, self.W.T)   # da el gradient el hayerga3 ll layer el ablena
        
        return dL_dX
    
    def get_params(self):
        """Return weights and biases"""
        return [self.W, self.b]
    
    def get_grads(self):
        """Return gradients of weights and biases"""
        return [self.dL_dW, self.dL_db]
    
    def update_params(self, dW, db, learning_rate):
        """Update weights and biases"""
        self.W -= learning_rate * dW   # hena bn-apply SGD update 3ala el weights
        self.b -= learning_rate * db   # w nafs el kalam 3ala el bias