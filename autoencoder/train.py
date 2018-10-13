## CS294A/CS294W Programming Assignment Starter Code

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sampleIMAGES import sampleIMAGES
from displayData import displayData
from initializeParameters import initializeParameters
from sparseAutoencoderCost import sparseAutoencoderCost
from computeNumericalGradient import computeNumericalGradient

plt.ion()

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  programming assignment. You will need to complete the code in sampleIMAGES.m,
#  sparseAutoencoderCost.m and computeNumericalGradient.m. 
#  For the purpose of completing the assignment, you do not need to
#  change the code in this file. 
#
##======================================================================
## STEP 0: Here we provide the relevant parameters values that will
#  allow your sparse autoencoder to get good filters; you do not need to 
#  change the parameters below.

visibleSize = 8 * 8     # number of input units 
hiddenSize = 25         # number of hidden units 
sparsityParam = 0.01    # desired average activation of the hidden units.
_lambda = 0.0001        # weight decay parameter
beta = 3                # weight of sparsity penalty term

##======================================================================
## STEP 1: Implement sampleIMAGES
#
#  After implementing sampleIMAGES, the display_network command should
#  display a random sample of 200 patches from the dataset

patches = sampleIMAGES()
displayData(patches[:,np.random.randint(10000, size=100)].T)

#  Obtain random parameters theta
theta = initializeParameters(hiddenSize, visibleSize)

input('\nProgram paused. Press enter to continue.\n')

##======================================================================
## STEP 2: Implement sparseAutoencoderCost
#
#  You can implement all of the components (squared error cost, weight decay term,
#  sparsity penalty) in the cost function at once, but it may be easier to do 
#  it step-by-step and run gradient checking (see STEP 3) after each step.  We 
#  suggest implementing the sparseAutoencoderCost function using the following steps:
#
#  (a) Implement forward propagation in your neural network, and implement the 
#      squared error term of the cost function.  Implement backpropagation to 
#      compute the derivatives.   Then (using lambda=beta=0), run Gradient Checking 
#      to verify that the calculations corresponding to the squared error cost 
#      term are correct.
#
#  (b) Add in the weight decay term (in both the cost function and the derivative
#      calculations), then re-run Gradient Checking to verify correctness. 
#
#  (c) Add in the sparsity penalty term, then re-run Gradient Checking to 
#      verify correctness.
#
#  Feel free to change the training settings when debugging your
#  code.  (For example, reducing the training set size or 
#  number of hidden units may make your code run faster; and setting beta 
#  and/or lambda to zero may be helpful for debugging.)  However, in your 
#  final submission of the visualized weights, please use parameters we 
#  gave in Step 0 above.

cost, grad = sparseAutoencoderCost(theta, visibleSize, hiddenSize, _lambda, sparsityParam, beta, patches.T)

##======================================================================
## STEP 3: Gradient Checking
#
# Hint: If you are debugging your code, performing gradient checking on smaller models 
# and smaller training sets (e.g., using only 10 training examples and 1-2 hidden 
# units) may speed things up.

# # Short hand for cost function
# costFunc = lambda p: sparseAutoencoderCost(p, visibleSize, hiddenSize, _lambda, sparsityParam, beta, patches.T[0:10])

# cost, grad = costFunc(theta)
# numgrad = computeNumericalGradient(costFunc, theta)

# # Visually examine the two gradient computations.  The two columns
# # you get should be very similar. 
# print(grad)
# print(numgrad)
# print('The above two columns you get should be very similar.\n \
# (Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n')

# # Evaluate the norm of the difference between two solutions.  
# # If you have a correct implementation, and assuming you used EPSILON = 0.0001 
# # in computeNumericalGradient.m, then diff below should be less than 1e-9
# diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)

# print('If your backpropagation implementation is correct, then \n \
#         the relative difference will be small (less than 1e-9). \n \
#         \nRelative Difference: %g\n'%diff)

# input('\nProgram paused. Press enter to continue.\n')

##======================================================================
## STEP 4: After verifying that your implementation of
#  sparseAutoencoderCost is correct, You can start training your sparse
#  autoencoder with minFunc (L-BFGS).

#  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize)

costFunc = lambda p: sparseAutoencoderCost(p, visibleSize, hiddenSize, _lambda, sparsityParam, beta, patches.T)
res = minimize(costFunc, theta, method='L-BFGS-B', jac=True, options={'maxiter': 400})
theta = res.x

Theta1 = theta[0:hiddenSize * (visibleSize + 1)].reshape(hiddenSize, visibleSize + 1)

displayData(Theta1[:,1:])

input('\nProgram paused. Press enter to continue.\n')