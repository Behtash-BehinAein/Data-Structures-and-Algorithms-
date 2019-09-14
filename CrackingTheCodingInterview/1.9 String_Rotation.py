import os
clear = lambda: os.system('cls')
clear()


def isSubstring(s1,s2):
    if s1 in s2 or s2 in s1:
        return True
    else:
        return False  

def isStringRotation(s1,s2):
    if len(s1) != len(s2):
        return False 
    s1s1 = s1 + s1
    return isSubstring(s1s1,s2)


#==================================================================================


s1 =  'waterbottlt'
s2 =  'erbottlewat'
print(isStringRotation(s1,s2))