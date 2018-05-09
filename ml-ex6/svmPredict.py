import matplotlib.pyplot as plt
import numpy as np

#SVMPREDICT returns a vector of predictions using a trained SVM model
#(svmTrain). 
#   pred = SVMPREDICT(model, X) returns a vector of predictions using a 
#   trained SVM model (svmTrain). X is a mxn matrix where there each 
#   example is a row. model is a svm model returned from svmTrain.
#   predictions pred is a m x 1 column of predictions of {0, 1} values.
#
def svmPredict(model, X):
    
    # Check if we are getting a column vector, if so, then assume that we only
    # need to do prediction for a single example
    if X.shape[1] == 1:
        # Examples should be in rows
        X = X.T 

    # Dataset
    m = X.shape[0]
    p = np.zeros(m).reshape(-1,1)
    pred = np.zeros(m).reshape(-1,1)

    if model['kernelFunction'].__name__ == 'linearKernel':
        # We can use the weights and bias directly if working with the 
        # linear kernel
        p = np.dot(X, model['w']) + model['b']
    elif model['kernelFunction'].__name__ == 'gaussianKernel':
        # Vectorized RBF Kernel
        # This is equivalent to computing the kernel on every pair of examples
        X1 = np.sum(X*X, axis=1).reshape(-1,1)
        X2 = np.sum(model['X']*model['X'], axis=1).reshape(1,-1)
        K = X1 + (X2 - 2 * np.dot(X, model['X'].T))
        K = np.power(model['kernelFunction'](np.array(1), np.array(0)), K)
        K = model['y'].T * K
        K = model['alphas'].T * K
        p = np.sum(K, axis=1)
    else:
        # Other Non-linear kernel
        for i in range(m):
            prediction = 0
            for j in range(model['X'].shape[0]):
                prediction = prediction + \
                    np.dot(model['alphas'][j], model['y'][j]) * \
                    model['kernelFunction'](X[i,:].T, model['X'][j,:].T)
            p[i] = prediction + model['b']

    # Convert predictions into 0 / 1
    pred[np.where(p >= 0)] = 1
    pred[np.where(p < 0)] = 0

    return pred

