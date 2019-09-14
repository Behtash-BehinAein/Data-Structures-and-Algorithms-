import os
clear = lambda: os.system('cls')
clear()


def OneAway(s, t):


    if  abs(len(s) - len(t)) >1:
        return False
    
    if len(s) < len(t):   # Swap to have "t" as the shorter string all the time 
        s, t = t, s

    for ii in range(len(t)):
        if t[ii] != s[ii]:
            print('Yes')
            return t[ii+1:] == s[ii+1:] or t[ii:] == s[ii+1:]   
    
    return len(s) != len(t)


s = 'teacher'
t = 'acher'

s = 'teacher'
t = 'detcher'

s = 'cab'
t = 'ad'

s = 'a'
t = 'ac'

s = 'ab'
t = 'acb'

s = 'A'
t = 'a'

s = 'a'
t = ''

s = ''
t = ''

print(OneAway(s, t))