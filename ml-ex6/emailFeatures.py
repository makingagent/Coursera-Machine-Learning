import numpy as np

#EMAILFEATURES takes in a word_indices vector and produces a feature vector
#from the word indices
#   x = EMAILFEATURES(word_indices) takes in a word_indices vector and 
#   produces a feature vector from the word indices. 
def emailFeatures(word_indices):

    # Total number of words in the dictionary
    n = 1899

    x = np.zeros(n).reshape(-1,1)

    for i in word_indices:
        x[i] = 1

    return x