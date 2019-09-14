import os
clear = lambda: os.system('cls')
clear()


# Approach 1
# ==========================================
# Brute Force O(n^2) method 
'''
def IsUnique(s):
    for ii in range(len(s)):
        char = s[ii]
        for jj in range(len(s)):
            if jj != ii:
                if s[jj] == char:
                    return False
    return True    
'''
# ==========================================

# Approach 2
# ==========================================
# Using a hash table O(n)
'''
def IsUnique(string):
    str_dct = {}
    str_len = len(string)
    for cc in string:
        str_dct[cc] = cc
    return str_len == len(str_dct)
'''
# ==========================================


# Approach 3
# ==========================================
# Using a boolean array and ascii codes
def IsUnique(string):

    string = string.replace(' ', '')   # Should ask if the space is included or not 
    ascii_cnt = [0]*128
    for char in string:
        if ascii_cnt[ord(char)] == 1:
            return False 
        else:  
            ascii_cnt[ord(char)] = 1
    return True 
# ==========================================



# Approach 4
# ==========================================
# Using sort O(nlogn)
'''
def IsUnique(s):
    s_l = list(s)
    s_l.sort()
    for ii in range(len(s_l)-1):
        if s_l[ii] == s_l[ii+1]:
            return False
    return True 
 '''   
# ==========================================


s1  = 'abcdefghij&A'
Answer  = IsUnique(s1)
print(Answer)







