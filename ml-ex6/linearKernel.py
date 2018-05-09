import numpy as np

#LINEARKERNEL returns a linear kernel between x1 and x2
#   sim = linearKernel(x1, x2) returns a linear kernel between x1 and x2
#   and returns the value in sim
def linearKernel(x1, x2):

    # Ensure that x1 and x2 are column vectors
    x1.reshape(1,-1)
    x2.reshape(1,-1)

    # Compute the kernel
    return np.dot(x1.T, x2)
