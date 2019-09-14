import os
clear = lambda: os.system('cls')
clear()

import numpy as np


def isPermutationOfPalindrome(string):
    string = string.lower()
    string = string.replace(' ', '')

    ascii_cnt = [0]*128
    for char in string:
        ascii_cnt[ord(char)] += 1 

    return sum(np.array(ascii_cnt)%2) <= 1 


#s = 'Tact Coa'
s = 'Able was I ere I saw Elba'
s = 'Never odd or even'
s = 'Adam'
print(isPermutationOfPalindrome(s))