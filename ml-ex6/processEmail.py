import numpy as np
import re
from getVocabList import getVocabList
from stemmer import *

#PROCESSEMAIL preprocesses a the body of an email and
#returns a list of word_indices 
#   word_indices = PROCESSEMAIL(email_contents) preprocesses 
#   the body of an email and returns a list of indices of the 
#   words contained in the email. 
#
def processEmail(email_contents):

    # Load Vocabulary
    vocabList = getVocabList()
    
    # Init return value
    word_indices = []

    # ========================== Preprocess Email ===========================

    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers

    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.sub(r"<[^<>]+>", " ", email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub(r"[0-9]+", "number", email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.sub(r"(http|https)://[^\s]*", "httpaddr", email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub(r"[^\s]+@[^\s]+", "emailaddr", email_contents)

    # Handle $ sign
    email_contents = re.sub(r"[$]+", "dollar", email_contents)


    # ========================== Tokenize Email ===========================

    # Output the email to screen as well
    print('\n==== Processed Email ====\n\n')

    # Process file
    l = 0

    strs = re.split(r'[ `\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?\n\t]', email_contents)

    p = PorterStemmer()
    for _str in strs:

        # Remove any non alphanumeric characters
        _str = re.sub(r"[^a-zA-Z0-9]", "", _str)

        # Stem the word 
        # (the porterStemmer sometimes has issues, so we use a try catch block)
        _str = p.stem(_str)
        
        # Skip the word if it is too short
        if len(_str) < 1:
            continue

        for i in range(len(vocabList)):
            if _str == vocabList[i]:
                word_indices.append(i)
                break

        if l + len(_str) + 1 > 78:
            print('')
            l = 0

        print(_str+" ", end='')

    # Print footer
    print('\n\n=========================\n')
    return word_indices

        
