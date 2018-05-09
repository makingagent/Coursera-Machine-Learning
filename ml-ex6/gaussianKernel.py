import numpy as np

#RBFKERNEL returns a radial basis function kernel between x1 and x2
#   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
#   and returns the value in sim
def gaussianKernel(x1, x2, sigma):

    # Ensure that x1 and x2 are column vectors
    x1.reshape(-1,1)
    x2.reshape(-1,1)

    # Compute the kernel
    return np.exp(-np.sum((x1-x2)*(x1-x2))/2/sigma/sigma)
