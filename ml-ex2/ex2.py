## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions 
#  in this exericse:
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
from costFunction import costFunction
from plotDecisionBoundary import plotDecisionBoundary
from sigmoid import sigmoid
from predict import predict

plt.ion()

## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]

## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')

plotData(X, y)

# Put some labels 
# Labels and Legend
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

# Specified in plot order
plt.legend(['Admitted', 'Not admitted'])

input('\nProgram paused. Press enter to continue.\n')


## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in 
#  costFunction.py

#  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = X.shape

# Add intercept term to x and X_test
X = np.vstack((np.ones(m), X.T)).T
y = y.reshape(-1,1)

# Initialize fitting parameters
theta = np.zeros(n+1)
# Compute and display initial cost and gradient
cost, grad = costFunction(theta, X, y)

print('Cost at initial theta (zeros): %f\n'%cost)
print('Gradient at initial theta (zeros): \n')
print(grad)

input('\nProgram paused. Press enter to continue.\n')


## ============= Part 3: Optimizing using minimize  =============
#  In this exercise, you will use a scipy function (minimize) to find the
#  optimal parameters theta.

#  Set options for minimize
res = minimize(costFunction, theta, method='BFGS', jac=True, options={'maxiter': 400}, args=(X, y))

# Print theta to screen
print('Cost at theta found by fminunc: %f\n'%res.fun)
print('theta: \n')
print(res.x)

# Plot Boundary
theta = res.x.reshape(-1,1)
plotDecisionBoundary(theta, X, y)

# Put some labels
# Labels and Legend
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

input('\nProgram paused. Press enter to continue.\n')

## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and 
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of 
#  our model.
#
#  Your task is to complete the code in predict.m

#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 

prob = sigmoid(np.dot([1, 45, 85], theta))
print('For a student with scores 45 and 85, we predict an admission probability of %f\n\n'%prob[0])

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: %.2f\n'%(np.mean(np.double(p == y)) * 100))

input('\nProgram paused. Press enter to continue.\n')