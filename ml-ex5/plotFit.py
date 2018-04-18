import numpy as np
from polyFeatures import polyFeatures
import matplotlib.pyplot as plt

#PLOTFIT Plots a learned polynomial regression fit over an existing figure.
#Also works with linear regression.
#   PLOTFIT(min_x, max_x, mu, sigma, theta, p) plots the learned polynomial
#   fit with power p and feature normalization (mu, sigma).
def plotFit(min_x, max_x, mu, sigma, theta, p):
    
    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.arange(min_x - 15, max_x + 25, 0.05).reshape(-1,1)

    # Map the X values 
    X_poly = polyFeatures(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly / sigma

    # Add ones
    m = X_poly.shape[0]
    X_poly = np.vstack((np.ones(m), X_poly.T)).T

    # Plot
    theta = theta.reshape(-1,1)
    plt.plot(x, np.dot(X_poly, theta), '--', linewidth=2)
    plt.axis([-70, 70, -50, 50])