import numpy as np
from sigmoid import sigmoid

# MAPFEATURE Feature mapping function to polynomial features
#
#   MAPFEATURE(X1, X2) maps the two input features
#   to quadratic features used in the regularization exercise.
#
#   Returns a new feature array with more features, comprising of 
#   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
#
#   Inputs X1, X2 must be the same size
#
def mapFeature(X1, X2):

    degree = 6
    out = np.ones(X1.shape)
    for i in range(1,degree+1):
        for j in range(0,i+1):
            out = np.row_stack((out, np.power(X1,(i-j))*np.power(X2,j)))

    return out