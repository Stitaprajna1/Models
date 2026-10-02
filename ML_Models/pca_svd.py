import numpy as np



class PCA:
    def __init__(self, n_components) -> None:
        self.n_components = n_components
        
    def fit(self, x):
        n_rows = x.shape[0]
        x_mean = np.mean(x, axis=0)
        x_scaled = x_mean - x
        U, S, Vt = np.linalg.svd(x_scaled, full_matrices=False)
        self.x_scaled = x_scaled
        self.U = U
        self.S = S
        self.components_ = Vt.T[:self.n_components]
        self.eigenvalues = S**2/(n_rows-1)
        self.explained_variance_ratio_ = self.eigenvalues/np.sum(self.eigenvalues)
        
    def transform(self):
        x_transformed = self.x_scaled @ self.components_
        return x_transformed
    
    def fit_transform(self, x):
        self.fit(x)
        x_transformed = self.transform()
        return x_transformed
    
# if __name__ == '__main__':
#     x = np.array([[3, 4], [2, 8], [6, 9]])
#     pca = PCA(n_components=2)
#     print(pca.fit_transform(x))
#     print(pca.explained_variance_ratio_)