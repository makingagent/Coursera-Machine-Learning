import numpy as np

#PCA Run principal component analysis on the dataset X
#   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
#   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
#
def pca(X):
    
    # Useful values
    m = X.shape[0]
    n = X.shape[1]

    sigma = np.dot(X.T, X) / m
    U, S, v = np.linalg.svd(sigma, full_matrices=True)

    return U, S