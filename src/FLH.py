""" Follow-the-Leading-History Algorithm
- Evolving process of online learning
Reference
    -  Adaptive Algorithms for Online Decision Problems(https://www.researchgate.net/publication/220138797_Adaptive_Algorithms_for_Online_Decision_Problems)
    - Non-stationary Online Regression(https://arxiv.org/abs/2011.06957)
"""

class FLH:
    def __init__(self, alpha : float):
        self.alpha = alpha
        self.s_set = [] # Algorithms / Estimator indices 
        self.p_dist = [] # distribution over S set