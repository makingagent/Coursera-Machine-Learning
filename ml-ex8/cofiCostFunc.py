import numpy as np
import matplotlib.pyplot as plt

#COFICOSTFUNC Collaborative filtering cost function
#   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
#   num_features, lambda) returns the cost and gradient for the
#   collaborative filtering problem.
#
def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, _lambda):

    # Unfold the U and W matrices from params
    X = params[0:num_movies*num_features].reshape(num_movies, num_features)
    Theta = params[num_movies*num_features:].reshape(num_users, num_features)

    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    J = np.sum(R * np.power(np.dot(X, Theta.T) - Y, 2)) / 2 + \
        _lambda * np.sum(np.power(Theta, 2)) / 2 + \
        _lambda * np.sum(np.power(X, 2)) / 2

    for i in range(num_movies):
        idx = np.where(R[i,:] == 1)[0]
        tempTheta = Theta[idx,:]
        tempY = Y[i,idx]
        X_grad[i,:] = np.dot(np.dot(X[i,:],tempTheta.T)-tempY, tempTheta) + _lambda * X[i,:]

    for i in range(num_users):
        idx = np.where(R[:,i] == 1)[0]
        tempX = X[idx,:]
        tempY = Y[idx,i]
        Theta_grad[i,:] = np.dot(np.dot(tempX, Theta[i,:].T).T-tempY, tempX) + _lambda * Theta[i,:]

    return J, np.append(X_grad.flatten(), Theta_grad.flatten())
