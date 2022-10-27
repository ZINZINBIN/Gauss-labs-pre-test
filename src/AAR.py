''' Aggregating Algorithms for Regression(AAR)
Reference
    - Ridge Regression sketch : https://towardsdatascience.com/how-to-code-ridge-regression-from-scratch-4b3176e5837c
    - Online Ridge Regression Method Using Sliding Windows(https://www.researchgate.net/publication/236648230_Online_Ridge_Regression_Method_Using_Sliding_Windows)
    - Github : https://github.com/daaltces/pydaal-getting-started/blob/master/4-interactive-tutorials/Regression_online_example.ipynb
'''
import numpy as np
from typing import Union
from tqdm.auto import tqdm

class RidgeRegressor(object):
    def __init__(self, input_dims : int, gamma : float = 0.01):
        self.input_dims = input_dims
        self.gamma = gamma
        self.A = gamma * np.ones((input_dims,input_dims))
        self.B = np.zeros((input_dims, 1))
        self.A_inv = np.ones((input_dims, input_dims)) * gamma
    
    def _update(self, x : np.ndarray, y : np.float64):
        assert x.shape == (self.input_dims,1), "input data x should have ({},1) shape".format(self.input_dims)   
        self.A = self.A + np.matmul(x.T,x)
        self.B = self.B + y * x
        
    def _update_inv(self, x : np.ndarray):
        assert x.shape == (self.input_dims,1), "input data x should have ({},1) shape".format(self.input_dims)  
        dA_inv = np.matmul(np.matmul(self.A_inv, x), np.matmul(self.A_inv, x).T)
        dA_inv /= np.matmul(x.T,np.matmul(self.A_inv,x)) + 1
        self.A_inv = self.A_inv - dA_inv
        
    def fit(self, x : np.ndarray, y : Union[np.ndarray, np.array]):
        m,n = x.shape
        for t in tqdm(range(0,m)):
            x_new = x[t,:].reshape(-1,1)
            if type(y) == np.array:
                y_new = y[t]
            else:
                y_new = y[t,:].reshape(-1,)
            self._update(x_new, y_new)
            self._update_inv(x_new)
           
    def predict(self, x : np.ndarray):
        m,n = x.shape
        y_hat = []
        for t in range(0,m):
            x_new = x[t,:].reshape(-1,1)
            pred = np.matmul(np.matmul(self.B.T, self.A_inv), x_new).item(0)
            y_hat.append(pred)
        return np.array(y_hat)

class ARR(RidgeRegressor):
    def __init__(self, input_dims : int, gamma : float = 0.01):
        super().__init__(input_dims, gamma)
        
    def _compute_inv_A(self, x : np.ndarray):
        assert x.shape == (self.input_dims,1), "input data x should have ({},1) shape".format(self.input_dims)  
        dA_inv = np.matmul(np.matmul(self.A_inv, x), np.matmul(self.A_inv, x).T)
        dA_inv /= np.matmul(x.T,np.matmul(self.A_inv,x)) + 1
        return self.A_inv - dA_inv

    def predict(self, x : np.ndarray):
        m,n = x.shape
        y_hat = []
        for t in range(0,m):
            x_new = x[t,:].reshape(-1,1)
            
            # use inverse matrix A with x_new
            inv_A = self._compute_inv_A(x_new)
            
            # Prediction
            pred = np.matmul(np.matmul(self.B.T, inv_A), x_new).item(0)
            y_hat.append(pred)
            
        return np.array(y_hat)