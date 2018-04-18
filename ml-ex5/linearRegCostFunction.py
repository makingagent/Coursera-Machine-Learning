import numpy as np

#LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
#regression with multiple variables
#   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
#   cost of using theta as the parameter for linear regression to fit the 
#   data points in X and y. Returns the cost in J and the gradient in grad
def linearRegCostFunction(X, y, theta, _lambda):
    
    # Initialize some useful values
    m = len(y) # number of training examples

    theta = theta.reshape(-1,1)
    _theta = np.copy(theta) # !!! important, don't use reference
    _theta[0,0] = 0

    error = np.dot(X,theta) - y
    J = np.dot(error.T, error) / m/2 + _lambda * np.dot(_theta.T, _theta) /m/2
    grad = np.dot(X.T, error) / m + _lambda * _theta / m

    return J, grad.reshape(1,-1)[0]
    