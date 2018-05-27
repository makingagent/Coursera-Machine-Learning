import numpy as np

#RECOVERDATA Recovers an approximation of the original data when using the 
#projected data
#   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
#   original data that has been reduced to K dimensions. It returns the
#   approximate reconstruction in X_rec.
#
def recoverData(Z, U, K):
    
    # You need to return the following variables correctly.
    X_rec = np.zeros((Z.shape[0], U.shape[0]))

    for i in range(Z.shape[0]):
        v = Z[i, :]
        for j in range(U.shape[0]):
            X_rec[i, j] = np.dot(v, U[j, 0:K].T)


    return X_rec