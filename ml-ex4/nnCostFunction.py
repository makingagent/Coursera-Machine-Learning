import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient

#NNCOSTFUNCTION Implements the neural network cost function for a two layer
#neural network which performs classification
#   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
#   X, y, lambda) computes the cost and gradient of the neural network. The
#   parameters for the neural network are "unrolled" into the vector
#   nn_params and need to be converted back into the weight matrices. 
# 
#   The returned parameter grad should be a "unrolled" vector of the
#   partial derivatives of the neural network.
#
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda):
    
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape(\
                 hidden_layer_size, input_layer_size + 1)
    
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(\
                 num_labels, hidden_layer_size + 1)

    # Setup some useful variables
    m = len(y) # number of training examples

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the code by working through the
    #               following parts.
    #
    # Part 1: Feedforward the neural network and return the cost in the
    #         variable J. After implementing Part 1, you can verify that your
    #         cost function computation is correct by verifying the cost
    #         computed in ex4.m
    #
    # Part 2: Implement the backpropagation algorithm to compute the gradients
    #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    #         Theta2_grad, respectively. After implementing Part 2, you can check
    #         that your implementation is correct by running checkNNGradients
    #
    #         Note: The vector y passed into the function is a vector of labels
    #               containing values from 1..K. You need to map this vector into a 
    #               binary vector of 1's and 0's to be used with the neural network
    #               cost function.
    #
    #         Hint: We recommend implementing backpropagation using a for-loop
    #               over the training examples if you are implementing it for the 
    #               first time.
    #
    # Part 3: Implement regularization with the cost function and gradients.
    #
    #         Hint: You can implement this around the code for
    #               backpropagation. That is, you can compute the gradients for
    #               the regularization separately and then add them to Theta1_grad
    #               and Theta2_grad from Part 2.

    a1 = np.vstack((np.ones(m), X.T)).T
    a2 = sigmoid(np.dot(a1, Theta1.T))
    a2 = np.vstack((np.ones(m), a2.T)).T
    a3 = sigmoid(np.dot(a2, Theta2.T))
    y = np.tile((np.arange(num_labels)+1)%10,(m,1)) == np.tile(y,(1,num_labels))

    regTheta1 = Theta1[:,1:]
    regTheta2 = Theta2[:,1:]

    J = -np.sum( y * np.log(a3) + (1-y) * np.log(1-a3) ) / m + \
        _lambda * np.sum(regTheta1*regTheta1) / m/2 + \
        _lambda * np.sum(regTheta2*regTheta2) / m/2

    delta1 = np.zeros(Theta1.shape)
    delta2 = np.zeros(Theta2.shape)
    for i in range(m):
        a1_ = a1[i]; a2_ = a2[i]; a3_ = a3[i]
        d3 = a3_ - y[i]; d2 = np.dot(d3,Theta2) * sigmoidGradient(np.append(1,np.dot(a1_, Theta1.T)))
        delta1 = delta1 + np.dot(d2[1:].reshape(-1,1),a1_.reshape(1,-1)); 
        delta2 = delta2 + np.dot(d3.reshape(-1,1), a2_.reshape(1,-1))

    regTheta1 = np.vstack((np.zeros(Theta1.shape[0]), regTheta1.T)).T
    regTheta2 = np.vstack((np.zeros(Theta2.shape[0]), regTheta2.T)).T
    Theta1_grad = delta1 / m + _lambda * regTheta1 / m
    Theta2_grad = delta2 / m + _lambda * regTheta2 / m

    grad = np.append(Theta1_grad.flatten(), Theta2_grad.flatten())
    print('cost value: %lf'%J)
    
    return J, grad