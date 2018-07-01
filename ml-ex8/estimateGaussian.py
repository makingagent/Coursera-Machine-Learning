import numpy as np

#ESTIMATEGAUSSIAN This function estimates the parameters of a 
#Gaussian distribution using the data in X
#   [mu sigma2] = estimateGaussian(X), 
#   The input X is the dataset with each n-dimensional data point in one row
#   The output is an n-dimensional vector mu, the mean of the data set
#   and the variances sigma^2, an n x 1 vector
# 
def estimateGaussian(X):

    # Useful variables
    m, n = X.shape

    # You should return these values correctly
    mu = np.zeros(n)
    sigma = np.zeros(n)

    mu = np.sum(X, axis=0) / m
    sigma2 = np.sum(np.power(X-mu,2), axis=0) / m

    return mu, sigma2
