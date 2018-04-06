import matplotlib.pyplot as plt
import numpy as np
from lrCostFunction import lrCostFunction
from scipy.optimize import minimize


#ONEVSALL trains multiple logistic regression classifiers and returns all
#the classifiers in a matrix all_theta, where the i-th row of all_theta 
#corresponds to the classifier for label i
#   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
#   logisitc regression classifiers and returns each of these classifiers
#   in a matrix all_theta, where the i-th row of all_theta corresponds 
#   to the classifier for label i
def oneVsAll(X, y, num_labels, _lambda):

    # Some useful variables
    m, n = X.shape

    # You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    X = np.vstack((np.ones(m), X.T)).T
    y = y.reshape(-1,1)

    initial_theta = np.zeros(n + 1)

    for c in range(num_labels):

        res = minimize(lrCostFunction, initial_theta, method='CG', jac=True, options={'maxiter': 50}, args=(X, y==c, _lambda))
        all_theta[c] = res.x

    return all_theta