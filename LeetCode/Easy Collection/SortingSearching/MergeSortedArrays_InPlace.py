import os
clear = lambda: os.system('cls')
clear()



def merge(nums1, m, nums2, n):


    ll = m - 1
    rr = n - 1
    cc = m+n -1 
    while ll>=0 and rr >= 0:
        if nums1[ll] < nums2[rr]:
            nums1[cc] = nums2[rr]
            rr -= 1
        else: 
            nums1[cc] = nums1[ll]
            ll -= 1
        cc -=1 
    '''
    while ll>=0:                     
        nums1[cc] = nums1[ll]
        ll -= 1
        cc -= 1
    '''
    while rr>=0:                     
        nums1[cc] = nums2[rr]
        rr -= 1
        cc -= 1


'''
nums1 = [1,2,3,0,0,0] 
m = 3
# --------------------
nums2 = [2,5,6]       
n = 3
'''

nums1 = [2,0] 
m = 1
# --------------------
nums2 = [1]       
n = 1



merge(nums1, m, nums2, n)
print(nums1)
        



