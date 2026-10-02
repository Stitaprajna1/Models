import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.x_scaled = None
        self.eigenvalues = None
        self.eigenvectors = None

    def fit(self, x):
        x_mean = np.mean(x, axis=0)
        x_scaled = x - x_mean
        x_cov = np.cov(x_scaled)
        self.x_scaled = x_scaled 
        eigenvalues, eigenvectors = np.linalg.eig(x_cov)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

    def transform(self):
        x_transformed = self.eigenvectors.T @ self.x_scaled
        x_transformed_real = np.real(x_transformed)
        order = np.argsort(-self.eigenvalues)
        x_transformed_real = x_transformed_real[order]
        return x_transformed_real[:self.n_components]

    def fit_transform(self, x):
        #fit
        self.fit(x)
        #transform
        x_transformed = self.transform()
        return x_transformed
    
