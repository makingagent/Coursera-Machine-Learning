import numpy as np
from sigmoid import sigmoid

#PREDICT Predict the label for a trained one-vs-all classifier. The labels 
#are in the range 1..K, where K = size(all_theta, 1). 
#  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
#  for each example in the matrix X. Note that X contains the examples in
#  rows. all_theta is a matrix where the i-th row is a trained logistic
#  regression theta vector for the i-th class. You should set p to a vector
#  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
#  for 4 examples) 
def predictOneVsAll(all_theta, X):
    
    m = X.shape[0] # Number of training examples

    # Add ones to the X data matrix
    X = np.vstack((np.ones(m), X.T)).T

    return np.argmax(sigmoid(np.dot(all_theta, X.T)), axis=0)