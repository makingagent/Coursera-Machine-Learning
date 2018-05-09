import numpy as np

#SVMTRAIN Trains an SVM classifier using a simplified version of the SMO 
#algorithm. 
#   [model] = SVMTRAIN(X, Y, C, kernelFunction, tol, max_passes) trains an
#   SVM classifier and returns trained model. X is the matrix of training 
#   examples.  Each row is a training example, and the jth column holds the 
#   jth feature.  Y is a column matrix containing 1 for positive examples 
#   and 0 for negative examples.  C is the standard SVM regularization 
#   parameter.  tol is a tolerance value used for determining equality of 
#   floating point numbers. max_passes controls the number of iterations
#   over the dataset (without changes to alpha) before the algorithm quits.
#
# Note: This is a simplified version of the SMO algorithm for training
#       SVMs. In practice, if you want to train an SVM classifier, we
#       recommend using an optimized package such as:  
#
#           LIBSVM   (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
#           SVMLight (http://svmlight.joachims.org/)
#
#
def svmTrain(X, Y, C, kernelFunction, tol=None, max_passes=None):

    if tol == None:
        tol = 1e-3 

    if max_passes == None:
        max_passes = 5

    # Data parameters
    m = X.shape[0]
    n = X.shape[1]

    Y = Y.copy()
    Y[np.where(Y==0)] = -1

    # Variables
    alphas = np.zeros(m).reshape(-1,1)
    b = 0
    E = np.zeros(m).reshape(-1,1)
    passes = 0
    eta = 0
    L = 0
    H = 0

    # Pre-compute the Kernel Matrix since our dataset is small
    # (in practice, optimized SVM packages that handle large datasets
    #  gracefully will _not_ do this)
    # 
    # We have implemented optimized vectorized version of the Kernels here so
    # that the svm training will run faster.

    if kernelFunction.__name__ == 'linearKernel':
        # Vectorized computation for the Linear Kernel
        # This is equivalent to computing the kernel on every pair of examples
        K = np.dot(X, X.T)
    elif kernelFunction.__name__ == 'gaussianKernel':
        # Vectorized RBF Kernel
        # This is equivalent to computing the kernel on every pair of examples
        X2 = np.sum(X*X, axis=1).reshape(-1,1)
        K = X2 + (X2.T - 2 * np.dot(X, X.T))
        K = np.power(kernelFunction(np.array(1), np.array(0)), K)
    else:
        # Pre-compute the Kernel Matrix
        # The following can be slow due to the lack of vectorization
        K = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                K[i,j] = kernelFunction(X[i,:].T, X[j,:].T)
                K[j,i] = K[i,j] #the matrix is symmetric


    # Train
    print('\nTraining ...', end='')
    dots = 12
    while passes < max_passes:

        num_changed_alphas = 0
        for i in range(m):

            # Calculate Ei = f(x(i)) - y(i) using (2). 
            # E(i) = b + sum (X(i, :) * (repmat(alphas.*Y,1,n).*X)') - Y(i);
            E[i] = b + np.sum(alphas * Y * K[:,i].reshape(-1,1)) - Y[i]

            # print(np.zeros(m).reshape(-1,1))
            if (Y[i]*E[i] < -tol and alphas[i] < C) or (Y[i]*E[i] > tol and alphas[i] > 0):

                # In practice, there are many heuristics one can use to select
                # the i and j. In this simplified code, we select them randomly.
                j = np.int(np.ceil(m * np.random.rand()))-1
                while j == i: # Make sure i != j
                    j = np.int(np.ceil(m * np.random.rand()))-1

                # Calculate Ej = f(x(j)) - y(j) using (2).
                E[j] = b + np.sum(alphas * Y * K[:,j].reshape(-1,1)) - Y[j]

                # Save old alphas
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()

                # Compute L and H by (10) or (11). 
                if Y[i] == Y[j]:
                    L = np.maximum(0, alphas[j] + alphas[i] - C)
                    H = np.minimum(C, alphas[j] + alphas[i])
                else:
                    L = np.maximum(0, alphas[j] - alphas[i])
                    H = np.minimum(C, C + alphas[j] - alphas[i])

                if L == H:
                    # continue to next i. 
                    continue

                # Compute eta by (14).
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    # continue to next i. 
                    continue

                # Compute and clip new value for alpha j using (12) and (15).
                alphas[j] = alphas[j] - (Y[j] * (E[i] - E[j])) / eta

                # Clip
                alphas[j] = np.minimum(H, alphas[j])
                alphas[j] = np.maximum(L, alphas[j])

                # Check if change in alpha is significant
                if abs(alphas[j] - alpha_j_old) < tol:
                    # continue to next i.
                    # replace anyway
                    alphas[j] = alpha_j_old
                    continue

                # Determine value for alpha i using (16).
                alphas[i] = alphas[i] + Y[i]*Y[j]*(alpha_j_old - alphas[j])

                # Compute b1 and b2 using (17) and (18) respectively.
                b1 = b - E[i] \
                        - Y[i] * (alphas[i] - alpha_i_old) * K[i,j] \
                        - Y[j] * (alphas[j] - alpha_j_old) * K[i,j]
                b2 = b - E[j] \
                        - Y[i] * (alphas[i] - alpha_i_old) * K[i,j] \
                        - Y[j] * (alphas[j] - alpha_j_old) * K[j,j]

                # Compute b by (19).
                if 0 < alphas[i] and alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] and alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2)/2

                num_changed_alphas = num_changed_alphas + 1
                
        if num_changed_alphas == 0:
            passes = passes + 1
        else:
            passes = 0

        print('.', end='')
        dots = dots + 1
        if dots > 78:
            dots = 0
            print('')
    
    print(' Done! \n\n')
    idx = np.where(alphas > 0)
    return {
        'X': X[idx[0],:],
        'y': Y[idx[0]],
        'kernelFunction': kernelFunction,
        'b': b,
        'alphas': alphas[idx[0]],
        'w': np.dot((alphas*Y).T,X).T
    }





