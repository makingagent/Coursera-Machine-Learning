import numpy as np
import matplotlib.pyplot as plt

#GETMOVIELIST reads the fixed movie list in movie.txt and returns a
#cell array of the words
#   movieList = GETMOVIELIST() reads the fixed movie list in movie.txt 
#   and returns a cell array of the words in movieList.
def loadMovieList():

    ## Read the fixed movieulary list
    f = open('movie_ids.txt', 'r')

    # Store all movies in cell array movie{}
    n = 1682  # Total number of movies 

    movieList = [1]*n
    for i in range(n):
        line = f.readline()
        movieList[i] = line[line.find(' ')+1:]
    f.close()

    return movieList
