import os
clear = lambda: os.system('cls')
clear()



def isBadVersion(n):
    arr = [False, False, False, True, True]
    if arr[n-1] == True:
        return True
    else: 
        return False



def firstBadVersion(n):
    """
    :type n: int
    :rtype: int
    """

    ll = 1
    rr = n

    while ll<rr:
        mid = ll + (rr-ll)//2
        if isBadVersion(mid) == True: 
            rr = mid   
            print(mid, 'rr:', rr)
        else: 
            ll = mid+1
            print(mid, 'll:', ll)

 
    return ll




  
arr = [False, False, False, True, True]


fbv = firstBadVersion(5)
print(fbv)

