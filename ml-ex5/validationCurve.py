import numpy as np
from trainLinearReg import trainLinearReg
from linearRegCostFunction import linearRegCostFunction

#VALIDATIONCURVE Generate the train and validation errors needed to
#plot a validation curve that we can use to select lambda
#   [lambda_vec, error_train, error_val] = ...
#       VALIDATIONCURVE(X, y, Xval, yval) returns the train
#       and validation errors (in error_train, error_val)
#       for different values of lambda. You are given the training set (X,
#       y) and validation set (Xval, yval).
#
def validationCurve(X, y, Xval, yval):
    
    # Selected values of lambda (you should not change this)
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]).reshape(-1,1)

    # You need to return these variables correctly.
    error_train = np.zeros(lambda_vec.shape[0])
    error_val = np.zeros(lambda_vec.shape[0])

    for i in range(lambda_vec.shape[0]):
        _lambda = lambda_vec[i]
        theta = trainLinearReg(X, y, _lambda)
        error_train[i], _ = linearRegCostFunction(X, y, theta, 0)
        error_val[i], _ = linearRegCostFunction(Xval, yval, theta, 0)

    return lambda_vec, error_train, error_val
