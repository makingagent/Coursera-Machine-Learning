import numpy as np

#FINDCLOSESTCENTROIDS computes the centroid memberships for every example
#   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
#   in idx for a dataset X where each row is a single example. idx = m x 1 
#   vector of centroid assignments (i.e. each entry in range [1..K])
#
def findClosestCentroids(X, centroids):

    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly.
    idx = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        min = np.inf 
        for j in range(K):
            diff = np.sum(np.power(X[i,:] - centroids[j,:], 2))
            if min > diff:
                min = diff
                idx[i] = j

    idx = idx.astype(int)

    return idx