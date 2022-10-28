""" Follow-the-Leading-History Algorithm
- Evolving process of online learning
Reference
    - Adaptive Algorithms for Online Decision Problems(https://www.researchgate.net/publication/220138797_Adaptive_Algorithms_for_Online_Decision_Problems)
    - Non-stationary Online Regression(https://arxiv.org/abs/2011.06957)
"""
import time
import numpy as np
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
        
        start_time = time.time()
        
        for t in tqdm(range(0,m), desc = "# Follow-the-Leading-History(FLH) : fit-predict process"):
            xt = x[t,:]
            yt = y[t]
            
            # Get new instance of estimator
            estimator = self.generate_estimator()
            
            # add estimators and update estimators with xt for j = 1,...,t
            self.estimators.append(estimator)
            
            for estimator in self.estimators:
                estimator.update_A(xt)           
            
            # prediction of the experts
            yt_hat = np.array([estimator(xt) for estimator in self.estimators]).reshape(-1,)
            
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
            
        end_time = time.time()
        
        dt = end_time - start_time
        
        print("# run time : {}".format(dt))
            
        return np.array(self.predictions)
    

class FLH_Revised(FLH):
    def __init__(self, alpha : float, gamma : float, input_dims : int, estimator = AAR):
        super().__init__(alpha, gamma, input_dims, estimator)
        self.St = [] # working set
        self.St_lifetime = []
    
    def _compute_lifetime(self, idx : int, k : int):
        r = idx // 2
        if r == 0:
            return np.power(2, k+2) + 1
        else:
            k += 1
            return self._compute_lifetime(r, k)
        
    def _update_St(self, t:int):
        self.St.append(t)
        lifetime = self._compute_lifetime(t, 0)
        self.St_lifetime.append(lifetime)
        
        for i in range(len(self.St_lifetime)):
            self.St_lifetime[i] -= 1
        
        remove_indices = []
        for i in range(len(self.St_lifetime)):
            if self.St_lifetime[i] <=0:
                remove_indices.append(i)
        
        if len(remove_indices) > 0:
            remove_idx = max(remove_indices) 
            
            self.St = self.St[remove_idx + 1:]
            self.St_lifetime = self.St_lifetime[remove_idx + 1:] 
            
            self.vts = self.vts[remove_idx + 1:]
                 
        else:
            pass 
        
    def fit_predict(self, x:np.ndarray, y:np.array):
        m,n = x.shape
        
        # initialize
        vt = 1
        st = 0
        st_lifetime = 5
        
        self.vts.append(vt)
        self.St.append(st)
        self.St_lifetime.append(st_lifetime)
        
        estimator = self.generate_estimator()
        self.estimators.append(estimator)
        
        start_time = time.time()
        
        for t in tqdm(range(0,m), desc = "# Follow-the-Leading-History revised(FLH) : fit-predict process"):
            xt = x[t,:]
            yt = y[t]
            
            for idx in self.St:
                self.estimators[idx].update_A(xt)      
            
            # prediction of the experts
            yt_hat = np.array([self.estimators[idx](xt) for idx in self.St]).reshape(-1,)
            
            # convex combination of the prediction
            vts = np.array(self.vts)
        
            pred = np.sum(vts * yt_hat)
            self.predictions.append(pred)
            
            # update estimators with yt
            for idx in self.St:
                self.estimators[idx].update_B(xt, yt)      
            
            # compute v_hat for i = t+1
            v_hat = self.compute_softmax(vts, yt, yt_hat)
            
            # update v_(t+1) for i = 1,2,...t
            v = (1 - 1 / (t+2)) * v_hat
            
            self.vts.clear()
            self.vts.extend(v.tolist())
            self.vts.append(1 / (t+2))
            
            # update estimator
            estimator = self.generate_estimator()
            self.estimators.append(estimator)
            
            # pruning step : update st to s(t+1)
            self._update_St(t+1)
            
        end_time = time.time()
        
        dt = end_time - start_time
        
        print("# run time : {}".format(dt))
            
        return np.array(self.predictions) 