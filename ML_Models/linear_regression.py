import numpy as np
import pandas as pd

# y = B.X
# B = VS-1U^T
# X = USV^T

class LinearRegressor:
    def __init__(self) -> None:
        self.coef_ = None
        self.intercept_ = None

    def fit(self, *args, **kwargs):
        """
        Train Linear Regression Model
        """
        X = args[0]
        y = args[1]

        # SVD
        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        V = Vt.T
        S_inv = np.diag(1/S)
        Ut = U.T
        y_mean = np.mean(y, axis=0)
        x_mean = np.mean(X, axis=0)
        
        # find slope and intercept
        b1 = V @ S_inv @ Ut @ y
        b0 = y_mean - x_mean @ b1

        self.coef_ = b1
        self.intercept_ = b0

    def predict(self, x):
        """
        predict values
        """
        b0, b1 = self.intercept_, self.coef_
        y = b0 + x @ b1
        return y
    

# if __name__ == '__main__':
#     x = np.random.default_rng().random(size=(10,2))
#     y = np.random.default_rng().uniform(low=0, high=5, size=(10,1))
#     model = LinearRegressor()
#     model.fit(x,y)
#     print(model.predict(x))
