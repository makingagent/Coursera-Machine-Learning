import numpy as np
from sigmoid import sigmoid

#COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
#   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
#   theta as the parameter for regularized logistic regression and the
#   gradient of the cost w.r.t. to the parameters. 
def costFunctionReg(theta, X, y, _lambda):
    
    # Initialize some useful values
    m = len(y) # number of training examples

    theta = theta.reshape(-1,1)
    _theta = np.copy(theta) # !!! important, don't use reference
    _theta[0,0] = 0

    s = sigmoid( np.dot(X,theta) )
    J = -( np.dot( y.T, np.log(s) ) + np.dot( (1-y).T, np.log(1-s) ) ) / m + _lambda * np.dot(_theta.T,_theta) / m/2

    grad = np.dot(X.T, ( s - y )) / m + _lambda * _theta / m

    return J[0], grad.reshape(1,-1)[0]