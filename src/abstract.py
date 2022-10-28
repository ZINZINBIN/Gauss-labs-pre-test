from abc import ABCMeta, abstractmethod

class Estimator(metaclass = ABCMeta):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def fit(self, x, y):
        pass
    
    @abstractmethod
    def predict(self, x):
        pass
    
    @abstractmethod
    def fit_predict(self, x):
        pass
    
    @abstractmethod
    def _update(self, x, y):
        pass
    
    @abstractmethod
    def _update_inv(self, x):
        pass