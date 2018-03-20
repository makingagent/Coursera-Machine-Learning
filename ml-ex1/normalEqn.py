import numpy as np

#NORMALEQN Computes the closed-form solution to linear regression 
#   NORMALEQN(X,y) computes the closed-form solution to linear 
#   regression using the normal equations.
def normalEqn(X, y):
    
    B = np.linalg.pinv(np.dot(X.T,X))
    theta = np.dot(np.dot(B,X.T), y)

    return theta