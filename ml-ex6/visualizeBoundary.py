import matplotlib.pyplot as plt
import numpy as np
from plotData import plotData
from svmPredict import svmPredict

#VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
#   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision 
#   boundary learned by the SVM and overlays the data on it
def visualizeBoundary(X, y, model):
    
    # Plot the training data on top of the boundary
    plotData(X,y.T[0])

    # Make classification predictions over a grid of values
    x1plot = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100).reshape(-1,1)
    x2plot = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100).reshape(-1,1)
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
        this_X = np.vstack((X1[:,i], X2[:,i])).T
        vals[:,i] = svmPredict(model, this_X).reshape(1,-1)

    # Plot the SVM boundary
    plt.contour(X1, X2, vals, colors='b', linewidths=1)