import os
clear = lambda: os.system('cls')
clear()


def CompressString(s):

    out = []
    cnt = 1
    for ii in range(1,len(s)):
        if s[ii] == s[ii-1]:
            cnt +=1
        else:
            out += s[ii-1] + str(cnt)
            cnt = 1
    out += s[ii-1] + str(cnt)
    
    if len(out) < len(s): return ''.join(out)        
    else: return s





s = 'aabcccccaaa'

print(CompressString(s))