import os
clear = lambda: os.system('cls')
clear()



# Approach 1
# ==========================================

# Using a new string and populating it 
def URLify(string, L):
    str_f = []
    for cc in range(L):
        if ord(string[cc]) != 32:
            str_f += string[cc]
        else: 
            str_f += '%20'
    return ''.join(str_f)

# =========================================


# Approach 2
# ==========================================
# In place using replace
'''
def URLify(string, L):
    count = 0
    for cc in range(L):
        if ord(string[cc]) == 32:
            count +=1 
    string = string.replace(' ','%20',count)
    return string
'''  
# ==========================================







s = 'Mr John Smith    '
L = 13
Answer  = URLify(s, L)
print(Answer)