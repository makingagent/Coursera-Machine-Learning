import numpy as np
import matplotlib.pyplot as plt

#PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
#index assignments in idx have the same color
#   PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those 
#   with the same index assignments in idx have the same color
def plotDataPoints(X, idx, K):

    # Create palette
    colors = plt.cm.hsv(idx / K)

    # Plot the data
    plt.scatter(X[:,0], X[:,1], facecolors='none', edgecolors=colors, marker='o', cmap='hsv')