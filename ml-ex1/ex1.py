## Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
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
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#

## Import
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from warmUpExercise import warmUpExercise
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent

## ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.py
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')
print(warmUpExercise())
print('\n')

input('Program paused. Press enter to continue.\n')

## ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
data = np.loadtxt('ex1data1.txt',delimiter=",")

X = data[:, 0]; y = data[:, 1]

m = len(y) # number of training examples

# Plot Data
# Note: You have to complete the code in plotData.py
plotData(X, y)

input('Program paused. Press enter to continue.\n')

## =================== Part 3: Gradient descent ===================
print('Running Gradient Descent ...\n')

X = np.vstack((np.ones(m), X)).T # Add a column of ones to x
y = y.reshape(-1,1)
theta = np.zeros((2, 1)) # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# compute and display initial cost
computeCost(X, y, theta)

# run gradient descent
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
print(theta)

# print theta to screen
print('Theta found by gradient descent: ')
print('%lf %lf \n'%(theta[0], theta[1]))

# Plot the linear fit
plt.plot(X[:,1], np.dot(X,theta), '-')
plt.legend(['Training data', 'Linear regression'])

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot(np.array([1, 3.5]),theta);
print('For population = 35,000, we predict a profit of %f\n'%(predict1*10000));
predict2 = np.dot(np.array([1, 7]),theta);
print('For population = 70,000, we predict a profit of %f\n'%(predict2*10000));

input('Program paused. Press enter to continue.\n')

## ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100);
theta1_vals = np.linspace(-1, 4, 100);

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([[theta0_vals[i]],[theta1_vals[j]]])
        J_vals[i][j] = computeCost(X, y, t)

# !!! important for plot
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

# Because of the way meshgrids work in the surf command, we need to 
# transpose J_vals before calling surf, or else the axes will be flipped
# Surface plo
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals.T,cmap='rainbow')
ax.set_xlabel(r'$\theta_0$'); ax.set_ylabel(r'$\theta_1$')

# Contour plot
plt.figure()
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
plt.contour(theta0_vals, theta1_vals, J_vals.T, levels=np.logspace(-2,3,20))
plt.xlabel(r'$\theta_0$'); plt.ylabel(r'$\theta_1$')
plt.plot(theta[0], theta[1], 'rx')

input('Program paused. Press enter to continue.\n')
plt.ioff()