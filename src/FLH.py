""" Follow-the-Leading-History Algorithm
- Evolving process of online learning
Reference
    -  Adaptive Algorithms for Online Decision Problems(https://www.researchgate.net/publication/220138797_Adaptive_Algorithms_for_Online_Decision_Problems)
    - Non-stationary Online Regression(https://arxiv.org/abs/2011.06957)
"""
import numpy as np
from src.abstract import Estimator
from src.AAR import AAR
from tqdm.auto import tqdm

class FLH:
    def __init__(self, alpha : float, gamma : float, input_dims : int, estimator = AAR):
        self.input_dims = input_dims
        self.alpha = alpha
        self.gamma = gamma
        
        self.vts_hat = []
        self.vts = []
        
        self.estimators = [] # online estimators group
        self._estimator_class = estimator
        
        self.predictions = []
    
    def generate_estimator(self):
        return self._estimator_class(self.input_dims, self.gamma)
        
    def compute_softmax(self, vt, yt, yt_hat):
        tot = np.sum(np.exp(-self.alpha * (yt - yt_hat)) * vt)
        vt_next = np.exp(-self.alpha * (yt - yt_hat)) * vt / tot
        return vt_next.reshape(-1,)
    
    def predict(self, x : np.ndarray):
        pass
        
    def fit_predict(self, x:np.ndarray, y:np.array):
        m,n = x.shape
        
        # initialize
        vt = 1
        self.vts.append(vt)
        
        for t in tqdm(range(0,m)):
            xt = x[t,:]
            yt = y[t]
            
            # Get new instance of estimator
            estimator = self.generate_estimator()
            
            # add estimators and update estimators with xt for j = 1,...,t
            self.estimators.append(estimator)
            
            for estimator in self.estimators:
                estimator.update_A(xt)           
            
            # prediction of the experts
            yt_hat = np.array([estimator(xt) for estimator in self.estimators])
            
            # convex combination of the prediction
            vts = np.array(self.vts)
        
            pred = np.sum(vts * yt_hat)
            self.predictions.append(pred)
            
            # update estimators with yt
            for estimator in self.estimators:
                estimator.update_B(xt, yt)    
            
            # compute v_hat for i = t+1
            v_hat = self.compute_softmax(vts, yt, yt_hat)
            
            # update v_(t+1) for i = 1,2,...t
            v = (1 - 1 / (t+2)) * v_hat
            
            self.vts.clear()
            self.vts.extend(v.tolist())
            self.vts.append(1 / (t+2))
            
        return np.array(self.predictions)