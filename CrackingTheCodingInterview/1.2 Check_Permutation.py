import os
clear = lambda: os.system('cls')
clear()


# (1) Sort both and compare one by one 
# ==========================================
'''
# Based on sorting both arrays O(nlogn)
def CheckPermutation(s1, s2):
    s1 = list(s1)
    s1.sort()
    s2 = list(s2)
    s2.sort()

    ii = 0
    while ii<len(s1):
        if s1[ii] != s2[ii]:
            return False
        ii += 1
    return True    
    '''
# ========================================== 
# (2) 3 Hash tables and their lengths 
# ==========================================
'''
# Based on 3 Hash tables O(n)
def CheckPermutation(s1, s2):

    s = list(s1) + list(s2)
    if len(list(s1)) != len(list(s2)):
        return False

    Dict1 = {}
    Dict2 = {}
    Dict3 = {}
    for ele in s1:
        Dict1[ele] = ele

    for ele in s2:
        Dict2[ele] = ele

    for ele in s:
        Dict3[ele] = ele

    if len(Dict1) == len(Dict2) and len(Dict1) == len(Dict3):
        return True
    else: 
        return False
'''     
# ==========================================

# (3) Character frequencies 
# ==========================================
def CheckPermutation(str1, str2):
    if len(str1) != len(str2):
        return False
    else: 
        ascii_cnt  = [0]*128
        for cc in range(len(str1)):
            ascii_cnt[ord(str1[cc])] +=1

        for cc in range(len(str2)):
            ascii_cnt[ord(str2[cc])] -=1
            if ascii_cnt[ord(str2[cc])] < 0:
                return False
    return True
# ==========================================



s1  = 'die'
s2  = 'eid'

Answer  = CheckPermutation(s1,s2)
print(Answer)







