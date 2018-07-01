import numpy as np

#MULTIVARIATEGAUSSIAN Computes the probability density function of the
#multivariate gaussian distribution.
#    p = MULTIVARIATEGAUSSIAN(X, mu, Sigma2) Computes the probability 
#    density function of the examples X under the multivariate gaussian 
#    distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
#    treated as the covariance matrix. If Sigma2 is a vector, it is treated
#    as the \sigma^2 values of the variances in each dimension (a diagonal
#    covariance matrix)
#
def multivariateGaussian(X, mu, Sigma2):

    k = mu.size

    if Sigma2.reshape(1, -1).shape[0] == 1 or Sigma2.reshape(1, -1).shape[1] == 1:
        Sigma2 = np.diag(Sigma2)

    X = X - mu
    p = np.power(2*np.pi, -k/2) * np.power(np.linalg.det(Sigma2), -0.5)
    p *= np.exp(-0.5 * np.sum(X.T * np.dot(np.linalg.pinv(Sigma2), X.T), axis=0))

    return p
