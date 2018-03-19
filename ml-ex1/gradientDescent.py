import numpy as np
from computeCost import computeCost

#GRADIENTDESCENT Performs gradient descent to learn theta
#   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
#   taking num_iters gradient steps with learning rate alpha
def gradientDescent(X, y, theta, alpha, num_iters):
    
    # Initialize some useful values
    m = len(y) # number of training examples
    J_history = np.zeros((num_iters, 1))

    for iter in range(num_iters):

        error = np.dot(X,theta) - y
        theta = theta - alpha * np.dot(X.T,error) / m

        J_history[iter][0] = computeCost(X, y, theta)

    return theta, J_history