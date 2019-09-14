import os
clear = lambda: os.system('cls')
clear()

# This alg uses merge sort 
def Sort_Array(arr):

    # Base Case
    if len(arr) <= 1:
        return 

    # Divide into 2 =======
    mid_idx  = len(arr)//2
    L_arr = arr[:mid_idx]
    R_arr = arr[mid_idx:]
    # =====================

    # Recursion ======================================
    # Keep dividing till you get to an elemental array
    Sort_Array(L_arr)
    Sort_Array(R_arr) 
    # ================================================

    # Build it back up from stacks ===================
    ll = 0
    rr = 0
    nn = 0
    # For every pairwise element of left and right arrays, copy the smaller one into main array 
    # This stops once either one of the arrays is exhausted 
    while ll < len(L_arr) and rr < len(R_arr): 
        if  L_arr[ll] < R_arr[rr]:
            arr[nn] = L_arr[ll]
            ll += 1
        else:
            arr[nn] = R_arr[rr]
            rr += 1
        nn += 1

    # Copy the rest (if any) of the left array into the main array 
    while ll < len(L_arr):
        arr[nn] = L_arr[ll]
        ll += 1
        nn += 1

    # Copy the rest (if any) of the right array into the main array 
    while rr < len(R_arr):
        arr[nn] = R_arr[rr]
        rr += 1
        nn += 1
    # ================================================

    return arr
# ==========================================
arr_1 = [200, 13, 34, 57, 23, 17, 18, 95, 61, 43, 22, 12, 3, 7, 8, 15, 32, 28, 24, 103, 100, 35]

Sorted_Array = Sort_Array(arr_1)
print(Sorted_Array)