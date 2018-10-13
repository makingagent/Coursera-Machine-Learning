import numpy as np

def initializeParameters(hiddenSize, visibleSize):
    
    # Initialize parameters randomly based on layer sizes.
    epsilon = np.sqrt(6) / np.sqrt(hiddenSize+visibleSize+1)
    W1 = np.random.rand(hiddenSize,visibleSize) * 2 * epsilon - epsilon
    W2 = np.random.rand(visibleSize,hiddenSize) * 2 * epsilon - epsilon
    W1 = np.vstack((np.zeros(W1.shape[0]), W1.T)).T
    W2 = np.vstack((np.zeros(W2.shape[0]), W2.T)).T

    return np.append(W1.flatten(), W2.flatten())