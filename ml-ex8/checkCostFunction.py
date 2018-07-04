import numpy as np
import matplotlib.pyplot as plt
from cofiCostFunc import cofiCostFunc
from computeNumericalGradient import computeNumericalGradient

#CHECKCOSTFUNCTION Creates a collaborative filering problem 
#to check your cost function and gradients
#   CHECKCOSTFUNCTION(lambda) Creates a collaborative filering problem 
#   to check your cost function and gradients, it will output the 
#   analytical gradients produced by your code and the numerical gradients 
#   (computed using computeNumericalGradient). These two gradient 
#   computations should result in very similar values.
def checkCostFunction(_lambda=None):

    if _lambda == None:
        _lambda = 0

    ## Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    # Zap out most entries
    Y = np.dot(X_t, Theta_t.T)
    Y[np.where(np.random.rand(Y.shape[0], Y.shape[1]) > 0.5)] = 0
    R = np.zeros(Y.shape)
    R[np.where(Y != 0)] = 1
    R = R.astype(int)

    ## Run Gradient Checking
    X = np.random.randn(X_t.shape[0], X_t.shape[1])
    Theta = np.random.randn(Theta_t.shape[0], Theta_t.shape[1])
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]

    func = lambda t: cofiCostFunc(t, Y, R, num_users, num_movies, num_features, _lambda)
    numgrad = computeNumericalGradient(func, np.append(X.flatten(), Theta.flatten()))
    
    cost, grad = cofiCostFunc(
        np.append(X.flatten(), Theta.flatten()), \
        Y, R, num_users, num_movies, num_features, _lambda
    )

    print(numgrad)
    print(grad)
    print("The above two columns you get should be very similar.\n \
        (Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n")

    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print('If your backpropagation implementation is correct, then \n \
        the relative difference will be small (less than 1e-9). \n \
        \nRelative Difference: %g\n'%diff)