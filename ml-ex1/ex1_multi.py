## Machine Learning Online Class
#  Exercise 1: Linear regression with multiple variables
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear regression exercise. 
#
#  You will need to complete the following functions in this 
#  exericse:
#
#     warmUpExercise.py
#     plotData.py
#     gradientDescent.py
#     computeCost.py
#     gradientDescentMulti.py
#     computeCostMulti.py
#     featureNormalize.py
#     normalEqn.py
#
#  For this part of the exercise, you will need to change some
#  parts of the code below for various experiments (e.g., changing
#  learning rates).
#

## Import
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from featureNormalize import featureNormalize
from normalEqn import normalEqn
from gradientDescentMulti import gradientDescentMulti

## ================ Part 1: Feature Normalization ================

print('Loading data ...\n')

## Load Data
data = np.loadtxt('ex1data2.txt',delimiter=",")
X = data[:, 0:2]
y = data[:, 2].reshape(-1,1)
m = len(y)

# Print out some data points
print('First 10 examples from the dataset: \n')
# no find function for python
# fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
print(X[0:10,:])
print(y[0:10])

input('Program paused. Press enter to continue.\n')

# Scale features and set them to zero mean
print('Normalizing Features ...\n')

X, mu, sigma = featureNormalize(X)

# Add intercept term to X
X = np.vstack((np.ones(m), X.T)).T


## ================ Part 2: Gradient Descent ================

print('Running gradient descent ...\n')

# Choose some alpha value
alpha = 0.01
num_iters = 400

# Init Theta and Run Gradient Descent 
theta = np.zeros((3, 1))
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.ion()
plt.figure()
plt.plot(np.arange(0,J_history.size,1), J_history, '-g')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')

# Display gradient descent's result
print('Theta computed from gradient descent: \n')
print(theta)
print('\n')

temp = np.array([[1.0, 1650.0, 3.0]])
temp[0,1:3] = (temp[0,1:3]-mu)/sigma
price = np.dot(temp, theta)

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f\n'%price)

input('Program paused. Press enter to continue.\n')

## ================ Part 3: Normal Equations ================

print('Solving with normal equations...\n')

## Load Data
data = np.loadtxt('ex1data2.txt',delimiter=",")
X = data[:, 0:2]
y = data[:, 2].reshape(-1,1)
m = len(y)

# Add intercept term to X
X = np.vstack((np.ones(m), X.T)).T

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations: \n')
print(theta)
print('\n')

temp = np.array([[1.0, 1650.0, 3.0]])
price = np.dot(temp, theta)

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f\n'%price)

input('Program paused. Press enter to continue.\n')