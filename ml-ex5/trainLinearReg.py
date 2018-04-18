import numpy as np
from linearRegCostFunction import linearRegCostFunction
from scipy.optimize import minimize

#TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
#regularization parameter lambda
#   [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
#   the dataset (X, y) and regularization parameter lambda. Returns the
#   trained parameters theta.
#
def trainLinearReg(X, y, _lambda):
    
    # Initialize Theta
    initial_theta = np.zeros(X.shape[1])

    # Create "short hand" for the cost function to be minimized
    costFunction = lambda t: linearRegCostFunction(X, y, t, _lambda)

    res = minimize(costFunction, initial_theta, method='BFGS', jac=True, tol=1e-2)
    print(res)

    return res.x