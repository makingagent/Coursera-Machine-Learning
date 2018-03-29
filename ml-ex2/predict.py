import numpy as np
from sigmoid import sigmoid

#PREDICT Predict whether the label is 0 or 1 using learned logistic 
#regression parameters theta
#   p = PREDICT(theta, X) computes the predictions for X using a 
#   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
def predict(theta, X):
    
    m = X.shape[0] # Number of training examples

    return np.round(sigmoid(np.dot(X, theta)))