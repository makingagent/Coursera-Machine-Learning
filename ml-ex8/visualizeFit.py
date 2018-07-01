import numpy as np
import matplotlib.pyplot as plt
from multivariateGaussian import multivariateGaussian

#VISUALIZEFIT Visualize the dataset and its estimated distribution.
#   VISUALIZEFIT(X, p, mu, sigma2) This visualization shows you the 
#   probability density function of the Gaussian distribution. Each example
#   has a location (x1, x2) that depends on its feature values.
#
def visualizeFit(X, mu, Sigma2):

    t = np.linspace(0, 35, 71)
    X1, X2 = np.meshgrid(t, t)
    
    Z = multivariateGaussian(np.vstack((X1.reshape(1,-1), X2.reshape(1,-1))).T, mu, Sigma2)
    Z = Z.reshape(X1.shape[0], -1)
    
    plt.figure()
    plt.plot(X[:, 0], X[:, 1],'bx', markersize=4)
    plt.axis([0, 30, 0, 30])

    # Do not plot if there are infinities
    if np.sum(np.isinf(Z)) == 0:
        plt.contour(X1, X2, Z, np.power(10, np.linspace(-20, 0, 7)))
