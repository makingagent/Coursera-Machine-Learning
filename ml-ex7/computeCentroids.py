import numpy as np

#COMPUTECENTROIDS returs the new centroids by computing the means of the 
#data points assigned to each centroid.
#   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
#   computing the means of the data points assigned to each centroid. It is
#   given a dataset X where each row is a single data point, a vector
#   idx of centroid assignments (i.e. each entry in range [1..K]) for each
#   example, and K, the number of centroids. You should return a matrix
#   centroids, where each row of centroids is the mean of the data points
#   assigned to it.
#
def computeCentroids(X, idx, K):

    # Useful variables
    m = X.shape[0]
    n = X.shape[1]

    # % You need to return the following variables correctly.
    centroids = np.zeros((K, n))

    num = np.zeros((K, 1))
    sum = np.zeros((K, n))

    for i in range(idx.shape[0]):
        z = idx[i]
        num[z] += 1
        sum[z,:] += X[i,:]

    centroids = sum / num

    return centroids