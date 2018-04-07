import numpy as np

#RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
#incoming connections and L_out outgoing connections
#   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
#   of a layer with L_in incoming connections and L_out outgoing 
#   connections. 
#
#   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
#   the column row of W handles the "bias" terms
#
def randInitializeWeights(L_in, L_out):
    
    epsilon = 0.12
    return np.random.rand(L_out,L_in+1) * 2 * epsilon - epsilon