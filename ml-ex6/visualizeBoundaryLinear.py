import matplotlib.pyplot as plt
import numpy as np
from plotData import plotData

#VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the
#SVM
#   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary 
#   learned by the SVM and overlays the data on it
def visualizeBoundaryLinear(X, y, model):
    
    w = model['w']
    b = model['b']
    xp = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100).reshape(-1,1)
    yp = -(w[0]*xp + b)/w[1]
    plotData(X, y.T[0])
    plt.plot(xp, yp, 'b-')
