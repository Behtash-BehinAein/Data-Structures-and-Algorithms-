import os
clear = lambda: os.system('cls')
clear()

# ==========================================
'''
# Using a hash table O(n)
def Contains_Duplicate(arr):
    idx = 0
    Dict = {}
    for ii in arr:
        Dict[ii] = idx
        idx +=1
        if len(Dict) != idx:
            return True
    return False
'''
# ==========================================

# ==========================================
# Using sort O(nlogn)
def Contains_Duplicate(arr):
    arr.sort()
    for ii in range(len(arr)-1):
        if arr[ii] == arr[ii+1]:
            return True
    return False 
# ==========================================




arr    = [1,2,3,4,5,6,7,8,9,10]
Answer  = Contains_Duplicate(arr)
print(Answer)







