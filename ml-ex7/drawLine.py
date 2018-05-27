import numpy as np
import matplotlib.pyplot as plt
from plotDataPoints import plotDataPoints

#DRAWLINE Draws a line from point p1 to point p2
#   DRAWLINE(p1, p2) Draws a line from point p1 to point p2 and holds the
#   current figure
def drawLine(p1, p2, *varargin):

    # Plot the examples
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], *varargin)
