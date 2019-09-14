import os
clear = lambda: os.system('cls')
clear()

import numpy as np


def OneAway(s1, s2):
    s1 = s1.lower().replace(' ', '')
    s2 = s2.lower().replace(' ', '')

    ascii_cnt = [0]*128
    for char in s1 :
        ascii_cnt[ord(char)] += 1 

    for char in s2 :
        ascii_cnt[ord(char)] -= 1 


    return sum(abs(np.array(ascii_cnt))) <=1


s1 = 'pale'
s2 = 'pale'

print(OneAway(s1, s2))