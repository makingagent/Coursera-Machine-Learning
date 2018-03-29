import numpy as np
from sigmoid import sigmoid

#COSTFUNCTION Compute cost and gradient for logistic regression
#   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
#   parameter for logistic regression and the gradient of the cost
#   w.r.t. to the parameters.
def costFunction(theta, X, y):
    
    # Initialize some useful values
    m = len(y) # number of training examples

    theta = theta.reshape(-1,1)

    s = sigmoid( np.dot(X,theta) )
    J = -( np.dot( y.T, np.log(s) ) + np.dot( (1-y).T, np.log(1-s) ) ) / m

    grad = np.dot(X.T, ( s - y )) / m

    return J[0], grad.reshape(1,-1)[0]