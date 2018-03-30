import matplotlib.pyplot as plt
import numpy as np

#PLOTDATA Plots the data points X and y into a new figure 
#   PLOTDATA(x,y) plots the data points with + for the positive examples
#   and o for the negative examples. X is assumed to be a Mx2 matrix.
def plotData(X, y):
    
    # Create New Figure
    plt.figure()
    # Find Indices of Positive and Negative Examples
    pos = np.where(y==1); neg = np.where(y==0)
    plt.plot(X[pos][:,0],X[pos][:,1], 'k+', linewidth=2, markersize=7)
    plt.plot(X[neg][:,0],X[neg][:,1], 'ko', markerfacecolor='y', markersize=7)
