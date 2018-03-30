## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.py
#     costFunction.py
#     predict.py
#     costFunctionReg.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from plotData import plotData
from mapFeature import mapFeature
from costFunctionReg import costFunctionReg
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict

plt.ion()

## Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).

data = np.loadtxt('ex2data2.txt', delimiter=',')
X = data[:, 0:2]; y = data[:, 2]

plotData(X, y)

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

# Specified in plot order
plt.legend(['y = 1', 'y = 0'])


## =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic 
#  regression to classify the data points. 
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = mapFeature(X[:,0], X[:,1]).T
y = y.reshape(-1,1)

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
_lambda = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost, grad = costFunctionReg(initial_theta, X, y, _lambda)

print('Cost at initial theta (zeros): %f\n'%cost)

input('\nProgram paused. Press enter to continue.\n')

## ============= Part 2: Regularization and Accuracies =============
#  Optional Exercise:
#  In this part, you will get to try different values of lambda and 
#  see how regularization affects the decision coundart
#
#  Try the following values of lambda (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?
#

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1 (you should vary this)
_lambda = 1

# Optimize
res = minimize(costFunctionReg, initial_theta, method='BFGS', jac=True, options={'maxiter': 400}, args=(X, y, _lambda))
theta = res.x.reshape(-1,1)

# Plot Boundary
plotDecisionBoundary(theta, X, y)

plt.title('lambda = %g'%_lambda)

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

plt.legend(['y = 1', 'y = 0', 'Decision boundary'])

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: %f\n'%(np.mean(np.double(p == y)) * 100))

input()