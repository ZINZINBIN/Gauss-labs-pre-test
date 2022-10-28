""" Follow-the-Leading-History Algorithm
- Evolving process of online learning
Reference
    -  Adaptive Algorithms for Online Decision Problems(https://www.researchgate.net/publication/220138797_Adaptive_Algorithms_for_Online_Decision_Problems)
    - Non-stationary Online Regression(https://arxiv.org/abs/2011.06957)
"""
import numpy as np
from src.abstract import Estimator

class FLH:
    def __init__(self, estimator : Estimator, alpha : float, gamma : float, input_dims : int):
        self.input_dims = input_dims
        self.alpha = alpha
        self.gamma = gamma
        self.s_set = [] # Algorithms / Estimator indices 
        self.p_dist = [] # distribution over S set
        self.estimators = [] # online estimators group
        self._estimator_class = estimator
        self.vt = []
    
    def generate_estimator(self):
        return self._estimator_class(self.input_dims, self.gamma)
        
    def compute_softmax(self, vt, yt, yt_hat):
        tot = np.sum(np.exp(-self.alpha * (yt - yt_hat)) * vt)
        vt_next = np.exp(-self.alpha * (yt - yt_hat)) * vt / tot
        return vt_next
    
    def predict(self, x : np.ndarray):
        pass
        
    def fit(self, x:np.ndarray, y:np.array):
        m,n = x.shape
        
        for t in range(0,m):
            
            xt = x[t,:]
            yt = y[t]
            
            # Get new instance of estimator
            estimator = self.generate_estimator()
            
            # add estimators and update estimators for j = 1,...,t
            self.estimators.append(estimator)
            
            yt_hat = np.array([estimator(xt) for estimator in self.estimators])
            v_hat = self.compute_softmax()
            