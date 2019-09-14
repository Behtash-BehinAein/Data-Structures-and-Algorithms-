import os
clear = lambda: os.system('cls')
clear()


def OneAway(s1, s2):
    ascii_cnt = [0]*128

    for char in s1 :
        ascii_cnt[ord(char)] += 1 

    for char in s2 :
        ascii_cnt[ord(char)] -= 1 
   
    cnt = 0
    for ele in ascii_cnt:
        if ele != 0:
            cnt += 1 

    return (cnt==2 and sum(ascii_cnt)==0) or cnt==0 



s = 'teacher'
t = 'acher'

s = ''
t = ''

s = 'cab'
t = 'ad'

s = 'a'
t = 'ac'

s = 'ab'
t = 'acb'

s = 'a'
t = ''


s = 'A'
t = 'a'

s = 'teacher'
t = 'detcher'

print(OneAway(s, t))