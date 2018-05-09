import matplotlib.pyplot as plt
import numpy as np
from plotData import plotData
from svmTrain import svmTrain
from gaussianKernel import gaussianKernel
from svmPredict import svmPredict

#EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
#where you select the optimal (C, sigma) learning parameters to use for SVM
#with RBF kernel
#   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
#   sigma. You should complete this function to return the optimal C and 
#   sigma based on a cross-validation set.
#
def dataset3Params(X, y, Xval, yval):
    
    choice = np.array([0.01,0.03,0.1,0.3,1,3,10,30]).reshape(-1,1)
    minError = np.inf
    curC = np.inf
    cur_sigma = np.inf

    for i in range(choice.shape[0]):
        for j in range(choice.shape[0]):
            func = lambda a, b: gaussianKernel(a, b, choice[j])
            func.__name__ = 'gaussianKernel'
            model = svmTrain(X, y, choice[i], func)
            predictions = svmPredict(model, Xval)
            error = np.mean(np.double(np.not_equal(predictions,yval)))
            if error < minError:
                minError = error
                curC = choice[i]
                cur_sigma = choice[j]


    C = curC
    sigma = cur_sigma

    return C, sigma
