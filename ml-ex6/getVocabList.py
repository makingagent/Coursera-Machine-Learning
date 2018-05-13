import numpy as np

#GETVOCABLIST reads the fixed vocabulary list in vocab.txt and returns a
#cell array of the words
#   vocabList = GETVOCABLIST() reads the fixed vocabulary list in vocab.txt 
#   and returns a cell array of the words in vocabList.
def getVocabList():

    ## Read the fixed vocabulary list
    f = open('vocab.txt', 'r')

    # Store all dictionary words in cell array vocab{}
    n = 1899 # Total number of words in the dictionary

    # For ease of implementation, we use a struct to map the strings => integers
    # In practice, you'll want to use some form of hashmap
    vocabList = [1]*n
    for i in range(n):
        # Word Index (can ignore since it will be = i)
        line = f.readline()
        vocabList[i] = line.split('\t')[1].strip()
    f.close()

    return vocabList