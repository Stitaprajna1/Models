import numpy as np
import pandas as pd


# z = b0 + Xb1
# p = 1/(1 + e^(-z))
# L = (y - y_p)^2
# b1(t) = b1(t-1) - a.dL/db1|x=xt
# b0(t) = b0(t-1) - a.dL/db0|x=xt

class LogisticRegression:

    def __init__(self, lr=0.00001, batch_size=8, epochs=100, threshold=0.5):
        self.coef_ = None
        self.intercept_ = 0
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epochs 
        self.threshold = threshold

    def CrossEntropyLoss(self, yp, y):
        n = len(y)
        loss = y @ np.log(yp) + (1 - y) @ np.log (1 - yp)
        loss = -np.sum(loss)/n
        return loss

    def GradiantDescend(self, y, yp, b0, b1, x):
        n = len(y)
        b1 = b1 - np.sum (x @ (y - yp), axis=0)/n
        b0 = b0 - np.sum((y - yp))/n
        return b0, b1
    
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def model(self, x):
        z = self.intercept_ + self.coef_ @ x
        logits = self.sigmoid(z)
        yp = (logits >= self.threshold).astype(int)
        return yp
    
    def build_batch(self):
        pass
        
    def fit(self, X, y, epochs):
        self.coef_ = np.zeros(X.shape[-1])
        dataset = self.build_batch(X, y)

        for epoch in epochs:
            for xb, yb in dataset:
                ybp = self.model(xb)
                self.intercept_, self.coef_ = self.GradiantDescend(
                                                    y = yb,
                                                    yp = ybp,
                                                    b0=self.intercept_,
                                                    b1 = self.coef_
                                                )



    def predict(self):
        
        pass