import numpy as np

#KMEANSINITCENTROIDS This function initializes K centroids that are to be 
#used in K-Means on the dataset X
#   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
#   used with the K-Means on the dataset X
#
def kMeansInitCentroids(X, K):

    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[0:K],:]

    return centroids