import numpy as np

#COMPUTECOST Compute cost for linear regression
#   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
#   parameter for linear regression to fit the data points in X and y
def computeCostMulti(X, y, theta):
    
    # Initialize some useful values
    m = len(y) # number of training examples

    error = np.dot(X,theta) - y
    J = (1 / (2*m)) * np.dot(error.T, error)

    return J