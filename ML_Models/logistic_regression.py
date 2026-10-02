import numpy as np
import pandas as pd


# z = b0 + Xb1
# p = 1/(1 + e^(-z))
# L = (y - y_p)^2
# b1(t) = b1(t-1) - a.dL/db1|x=xt
# b0(t) = b0(t-1) - a.dL/db0|x=xt

class LogisticRegressor:

    def __init__(self, lr=0.00001, batch_size=8, epochs=100, threshold=0.5):
        self.coef_ = None
        self.intercept_ = 0
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs 
        self.threshold = threshold

    def CrossEntropyLoss(self, yp, y):
        n = len(y)
        loss = y @ np.log(yp) + (1 - y) @ np.log (1 - yp)
        loss = -np.sum(loss)/n
        return loss

    def GradientDescent(self, y, yp, b0, b1, x):
        n = len(y)

        db1 = (x.T @ (y - yp))/n
        db0 = np.sum(y - yp)/n

        b1 = b1 - self.lr*db1
        b0 = b0 - self.lr*db0
        return b0, b1
    
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def model(self, x):
        z = self.intercept_ +  x @ self.coef_
        logits = self.sigmoid(z)
        return logits
    
    def build_batch(self, X, y):
        for i in range(0, len(X), self.batch_size):
            yield X[i:i+self.batch_size], y[i:i+self.batch_size]
        
    def fit(self, X, y):
        self.coef_ = np.zeros((X.shape[-1],1))

        for epoch in range(self.epochs):
            for xb, yb in self.build_batch(X, y):
                ybp = self.model(xb)
                self.intercept_, self.coef_ = self.GradientDescent(
                                                    y = yb,
                                                    yp = ybp,
                                                    b0 = self.intercept_,
                                                    b1 = self.coef_,
                                                    x = xb
                                                )

    def predict(self, x):
        logits = self.model(x)
        yp = (logits >= self.threshold).astype(int)
        return yp
    

# if __name__ == '__main__':
#     X = np.random.default_rng().random(size=(10,10))
#     y = np.random.default_rng().integers(low=0, high=2, size=(10,1))
#     model = LogisticRegressor()
#     model.fit(X,y)
#     print(X)
#     print('predicted:',model.predict(X))
#     print('actual', y)
#     print(model.coef_, model.intercept_)
    