import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.x_scaled = None
        self.eigenvalues = None
        self.eigenvectors = None

    def fit(self, x):
        x_mean = np.mean(x, axis=0)
        x_scaled = x_mean - x
        x_cov = np.cov(x_scaled, rowvar=False)
        self.x_scaled = x_scaled 
        eigenvalues, eigenvectors = np.linalg.eig(x_cov)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.explained_variance_ratio_ = np.sort(eigenvalues)[::-1]/np.sum(eigenvalues)

    def transform(self):
        x_transformed = self.x_scaled @ self.eigenvectors 
        x_transformed_real = np.real(x_transformed)
        order = np.argsort(-self.eigenvalues)
        x_transformed_real = x_transformed_real[:, order]
        return x_transformed_real[:, :self.n_components]

    def fit_transform(self, x):
        #fit
        self.fit(x)
        #transform
        x_transformed = self.transform()
        return x_transformed
    
# if __name__ == '__main__':
#     x = np.array([[3, 4], [2, 8], [6, 9]])
#     pca = PCA(n_components=2)
#     print(pca.fit_transform(x))
#     print(pca.explained_variance_ratio_)
