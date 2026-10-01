import numpy as np
import pandas as pd

# y = B.X
# B = VS-1U^T
# X = UZV^T

class LinearRegressor:
    def __init__(self) -> None:
        self.coef_ = None

    def fit(self, **kwargs):
        """
        Train Linear Regression Model
        """
        X = kwargs.get('X', [])
        y = kwargs.get('y', [])
        # intercept = kwargs.get('intercept', False)

        # SVD
        U, S, Vt = np.linalg.svd(X)

        V = Vt.T
        S_inv = np.diag(1/S)
        Xt = X.T
        y_mean = np.mean(y)
        xt_mean = np.mean(Xt)
        
        # find slope and intercept
        b1 = V @ S_inv @ U @ y
        b0 = y_mean - xt_mean @ b1

        self.coef_ = [b0, b1]

    def predict(self, x):
        """
        predict values
        """
        b0, b1 = self.coef_
        y = b0 + b1 @ x
        return y