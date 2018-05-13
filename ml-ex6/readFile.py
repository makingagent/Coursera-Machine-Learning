import numpy as np

#READFILE reads a file and returns its entire contents 
#   file_contents = READFILE(filename) reads a file and returns its entire
#   contents in file_contents
#
def readFile(filename):

    # Load File
    f = open(filename, 'r')
    if f:
        file_contents = f.read()
        f.close()
    else:
        file_contents = ''
        print('Unable to open %s\n', filename)

    return file_contents