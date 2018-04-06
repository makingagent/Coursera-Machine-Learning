import numpy as np
from sigmoid import sigmoid

#PREDICT Predict the label of an input given a trained neural network
#   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
#   trained weights of a neural network (Theta1, Theta2)
def predict(Theta1, Theta2, X):
    
    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    a1 = np.vstack((np.ones(m), X.T)).T

    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2)

    a2 = np.vstack((np.ones(m), a2.T)).T

    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)

    return np.argmax(a3, axis=1)