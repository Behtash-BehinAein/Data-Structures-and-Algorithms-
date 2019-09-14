import os
clear = lambda: os.system('cls')
clear()


# Recursion method =====================================
'''
# Sets an input parameter outside of the recursion loop 
def B_Search(lst, ele):
    return find_idx(lst, ele, 0)


def find_idx(lst, ele, idx):

    if len(lst) == 1:     # Base case
        if ele == lst[0]:
            return idx
        else: 
            print('Value is not in the list')
            return 

    mid = int(len(lst)/2)
    if ele >= lst[mid]:
        idx += mid
        lst = lst[mid:]
        return find_idx(lst, ele, idx)   # Recursion 

    else:
        lst = lst[:mid]
        return find_idx(lst, ele, idx) # Recursion
'''

# ====================================================

# ====================================================

# Index update method ================================
def B_Search(lst, ele):
    ll = 0 
    rr = len(lst) - 1
    while ll<=rr:
        mid = ll + (rr-ll)//2
        if lst[mid] == ele:
            return mid
        else:
            if ele > lst[mid]:
                ll = mid + 1
            else:
                rr = mid - 1 
# ====================================================








lst_1 = [21, 22, 24, 25, 27, 28, 33, 77, 99, 100, 200]
ele   = 25

INDEX  = B_Search(lst_1, ele)
print(INDEX)




