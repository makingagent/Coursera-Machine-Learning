import numpy as np

#SIGMOID Compute sigmoid functoon
#   J = SIGMOID(z) computes the sigmoid of z.
def sigmoid(z):
    
    return 1 / ( 1 + np.exp(-z) )