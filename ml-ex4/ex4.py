## Machine Learning Online Class - Exercise 4 Neural Network Learning

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     sigmoidGradient.m
#     randInitializeWeights.m
#     nnCostFunction.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#


## Initialization
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from displayData import displayData
from nnCostFunction import nnCostFunction
from sigmoidGradient import sigmoidGradient
from randInitializeWeights import randInitializeWeights
from checkNNGradients import checkNNGradients
from scipy.optimize import minimize
from predict import predict

plt.ion()

## Setup the parameters you will use for this exercise
input_layer_size  = 400   # 20x20 Input Images of Digits
hidden_layer_size = 25    # 25 hidden units
num_labels = 10           # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

data = sio.loadmat('ex4data1.mat')
X = data['X']; y = data['y']%10

m = X.shape[0]

# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[0:100]]

displayData(sel)

input('Program paused. Press enter to continue.\n')


## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta1 and Theta2
data = sio.loadmat('ex4weights.mat')
Theta1 = data['Theta1']; Theta2 = data['Theta2']

# Unroll parameters 
nn_params = np.append(Theta1, Theta2)

## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print('\nFeedforward Using Neural Network ...\n')

# Weight regularization parameter (we set this to 0 here).
_lambda = 0

J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, \
                   num_labels, X, y, _lambda)
                   
print('Cost at parameters (loaded from ex4weights): %f \
       \n(this value should be about 0.287629)\n'%(J))

input('\nProgram paused. Press enter to continue.\n')

## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

print('\nChecking Cost Function (w/ Regularization) ... \n')

# Weight regularization parameter (we set this to 1 here).
_lambda = 1

J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, \
                   num_labels, X, y, _lambda)

print('Cost at parameters (loaded from ex4weights): %f \
       \n(this value should be about 0.383770)\n'%(J))

input('\nProgram paused. Press enter to continue.\n')

## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

print('\nEvaluating sigmoid gradient...\n')

g = sigmoidGradient(np.array([1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ')
print(g)
print('\n\n')

input('\nProgram paused. Press enter to continue.\n')

## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)

print('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.append(initial_Theta1.flatten(), initial_Theta2.flatten())


## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.
#
print('\nChecking Backpropagation... \n')

#  Check gradients by running checkNNGradients
checkNNGradients()

input('\nProgram paused. Press enter to continue.\n')


## =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.
#

print('\nChecking Backpropagation (w/ Regularization) ... \n')

#  Check gradients by running checkNNGradients
_lambda = 3
checkNNGradients(_lambda)

# Also output the costFunction debugging values
debug_J, _  = nnCostFunction(nn_params, input_layer_size, \
                          hidden_layer_size, num_labels, X, y, _lambda)

print('\n\nCost at (fixed) debugging parameters (w/ lambda = 10): %f  \
         \n(this value should be about 0.576051)\n\n'%debug_J)

input('\nProgram paused. Press enter to continue.\n')


## =================== Part 8: Training NN ===================
#  You have now implemented all the code necessary to train a neural 
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#
print('\nTraining Neural Network... \n')

#  You should also try different values of lambda
_lambda = 1

# Create "short hand" for the cost function to be minimized
costFunction = lambda p: nnCostFunction(p, \
                                   input_layer_size, \
                                   hidden_layer_size, \
                                   num_labels, X, y, _lambda)

# Now, costFunction is a function that takes in only one argument (the
# neural network parameters)
res = minimize(costFunction, initial_nn_params, method='CG', jac=True, options={'maxiter': 200})
nn_params = res.x

# Obtain Theta1 and Theta2 back from nn_params
Theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape(\
                 hidden_layer_size, input_layer_size + 1)
    
Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(\
                 num_labels, hidden_layer_size + 1)

input('\nProgram paused. Press enter to continue.\n')

## ================= Part 9: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by 
#  displaying the hidden units to see what features they are capturing in 
#  the data.

print('\nVisualizing Neural Network... \n')

displayData(Theta1[:,1:])

input('\nProgram paused. Press enter to continue.\n')

## ================= Part 10: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = (predict(Theta1, Theta2, X)+1)%10

print('\nTraining Set Accuracy: %f\n'%(np.mean(np.double(pred == y.T)) * 100))

rp = np.random.permutation(m)

for i in range(m):
    # Display 
    print('\nDisplaying Example Image\n')
    t = np.array([X[rp[i]]])
    displayData(t)

    pred = predict(Theta1, Theta2, t)
    print('\nNeural Network Prediction: %d (digit %d)\n'%(pred, (pred+1)%10))

    input('Program paused. Press enter to continue.\n')