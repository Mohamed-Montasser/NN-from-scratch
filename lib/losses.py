# el file da ma3mol 3ashan ne7ot el loss functions el a7na besta3melha (MSE asasy fl project w CrossEntropy k option zyada)

import numpy as np

class MSE:
    """Mean Squared Error loss"""

    def __init__(self):
        self.Y_pred = None
        self.Y_true = None
    
    def forward(self, Y_true, Y_pred):
        self.Y_pred = Y_pred.reshape(Y_true.shape)   # hena bnet2aked en el shapes matshklsh moshkela fl 7esbat
        self.Y_true = Y_true
        
        error = self.Y_true - self.Y_pred           # el difference ben el true label w el prediction
        loss = np.mean(error ** 2)                  # bn7seb mse = average mngom3a el squared error
        
        return loss
    
    def backward(self):
        batch_size = self.Y_true.shape[0]           # 3adad el samples fl batch 3ashan n2assem 3alehom
        dL_dY = 2 * (self.Y_pred - self.Y_true) / batch_size  # derivative beta3 mse w.r.t el predictions
        return dL_dY
