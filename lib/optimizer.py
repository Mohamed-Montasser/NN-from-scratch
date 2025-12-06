# el file da feh el SGD optimizer elli benesta3melo 3ashan n-update el weights w el biases

import numpy as np

class SGD:
    """Stochastic Gradient Descent optimizer"""
    
    def __init__(self, learning_rate=0.01):
        """
        Initialize SGD optimizer
        
        Args:
            learning_rate: learning rate for parameter updates
        """
        self.learning_rate = learning_rate   # hena ben7afaz el learning rate 3ashan nesta3mlo fl updates
    
    def step(self, layers):
        """
        Update parameters using gradients
        
        W_new = W_old - learning_rate * dL/dW
        b_new = b_old - learning_rate * dL/db
        
        Args:
            layers: list of layer objects that have update_params method
        """
        for layer in layers:
            # Check if layer has trainable parameters
            if hasattr(layer, 'update_params'):     # bnet2aked en el layer da fe3lan leeh params tet-update
                grads = layer.get_grads()
                if len(grads) > 0:
                    dW = grads[0]                  # gradient 3ala el weights
                    db = grads[1]                  # gradient 3ala el biases
                    
                    layer.update_params(dW, db, self.learning_rate)  # hena ba-apply rule beta3 SGD 3ala el layer da
