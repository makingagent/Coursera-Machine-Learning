import numpy as np
import matplotlib.pyplot as plt

#SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
#outliers
#   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
#   threshold to use for selecting outliers based on the results from a
#   validation set (pval) and the ground truth (yval).
#
def selectThreshold(yval, pval):

    bestEpsilon = 0
    bestF1 = 0
    F1 = 0

    arr = np.linspace(np.amin(pval), np.amax(pval), 1000)

    for i in range(arr.size-2):

        epsilon = arr[i+1]

        predictions = pval < epsilon

        yval = yval.reshape(1,-1)[0]
        
        tp = np.sum((predictions == 1) & (yval == 1))
        fp = np.sum((predictions == 1) & (yval == 0))
        fn = np.sum((predictions == 0) & (yval == 1))
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        F1 = 2 * prec * rec /(prec + rec)

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

    return bestEpsilon, bestF1
