import numpy as np
import pandas as pd


# z = b0 + Xb1
# p = 1/(1 + e^(-z))
# L = (y - y_p)^2
# b1t = b1(t-1) - a.dL/db1|x=xt
# b0t = b0(t-1) - a.dL/db0|x=xt

class LogisticRegression:

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, epochs, lr):
        pass

    def predict(self):
        pass