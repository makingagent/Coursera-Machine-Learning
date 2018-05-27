import numpy as np

#PROJECTDATA Computes the reduced data representation when projecting only 
#on to the top k eigenvectors
#   Z = projectData(X, U, K) computes the projection of 
#   the normalized inputs X into the reduced dimensional space spanned by
#   the first K columns of U. It returns the projected examples in Z.
#
def projectData(X, U, K):
    
    # You need to return the following variables correctly.
    Z = np.zeros((X.shape[0], K))

    for i in range(X.shape[0]):
        x = X[i, :]
        Z[i, :] = np.dot(x, U[:,0:K])

    return Z