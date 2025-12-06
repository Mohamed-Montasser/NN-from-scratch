# el file da feh Sequential model elli ben-build 3aleh el networks beta3etna (xor, autoencoder, etc)
# bnedikh el layers bel tarteeb w howa by-handl el forward/backward w el training loop

import numpy as np

class Sequential:
    """Sequential neural network model"""
    
    def __init__(self, layers=None):
        """
        Initialize Sequential model
        
        Args:
            layers: list of layer objects (optional)
        """
        self.layers = layers if layers is not None else []   # hena bn-store el list beta3et el layers elli han-run 3aleha
    
    def add(self, layer):
        """Add a layer to the model"""
        self.layers.append(layer)    # function 3ashan nzawed layer gdida fl model
    
    def forward(self, X):
        """
        Forward pass through all layers
        
        Args:
            X: input data, shape (batch_size, input_size)
        
        Returns:
            Y: output after passing through all layers
        """
        Y = X
        for layer in self.layers:
            Y = layer.forward(Y)     # hena ben3ady 3ala kol layer b el output beta3 el layer elli ablaha
        return Y
    
    def backward(self, dL_dY):
        """
        Backward pass through all layers (reverse order)
        
        Args:
            dL_dY: initial gradient from loss function
        
        Returns:
            dL_dX: gradient w.r.t. input
        """
        for layer in reversed(self.layers):   # hena ben-loop 3aks el tarteeb 3ashan backprop
            dL_dY = layer.backward(dL_dY)     # kol layer berga3 el gradient ll layer elli ablaha
        return dL_dY
    
    def train_step(self, X, Y_true, loss_fn, optimizer):
        """
        Single training step: forward, backward, update
        
        Args:
            X: input batch
            Y_true: target batch
            loss_fn: loss function object
            optimizer: optimizer object
        
        Returns:
            loss: loss value for this batch
        """
        Y_pred = self.forward(X)                    # awel 7aga ben3mel forward pass
        
        loss = loss_fn.forward(Y_true, Y_pred)      # hena ben7seb el loss 3ala el batch
        
        dL_dY = loss_fn.backward()                  # dL/dY el gayy mn el loss function
        self.backward(dL_dY)                        # el backward 3ala el network kullaha
        
        optimizer.step(self.layers)                 # el optimizer by-update kol el trainable layers
        
        return loss
    
    def predict(self, X):
        """
        Make predictions (forward pass only)
        
        Args:
            X: input data
        
        Returns:
            Y: predicted output
        """
        return self.forward(X)          # fl testing aw evaluation benesta5dem forward bas
    
    def evaluate(self, X, Y_true, loss_fn):
        """
        Evaluate model on data
        
        Args:
            X: input data
            Y_true: target data
            loss_fn: loss function
        
        Returns:
            loss: average loss
        """
        Y_pred = self.forward(X)                     # forward pass 3ala el data
        loss = loss_fn.forward(Y_true, Y_pred)       # ben7seb el loss bas mesh ben3mel update
        return loss
    
    def get_trainable_layers(self):
        """Return list of trainable layers (Dense layers)"""
        trainable = []
        for layer in self.layers:
            if hasattr(layer, 'update_params'):      # bn-filter el layers elli fe3lan 3andaha parameters tet-update
                trainable.append(layer)
        return trainable
