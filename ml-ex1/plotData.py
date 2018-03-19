import matplotlib.pyplot as plt


#PLOTDATA Plots the data points x and y into a new figure 
#   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
#   population and profit.
def plotData(x, y):
    
    plt.ion()
    plt.figure()
    plt.plot(x, y, 'x')
    plt.axis([4, 24, -5, 25])
    plt.xlabel("Population of City in 10,000s") # setting the x label as population
    plt.ylabel("Profit in $10,000s") # setting the y label