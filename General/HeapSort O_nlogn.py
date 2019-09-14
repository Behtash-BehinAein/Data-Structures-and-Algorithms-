import os
clear = lambda: os.system('cls')
clear()


def Heapify(arr, n, idx):
   
    par_idx = idx

    L_child = 2*par_idx
    R_child = 2*par_idx + 1

    if L_child < n and arr[L_child] > arr[par_idx]:
        par_idx = L_child

    if R_child < n and arr[R_child] > arr[par_idx]:
        par_idx = R_child

    if par_idx != idx:   # Acts as the base case and breaks the recursion 
        arr[par_idx], arr[idx] = arr[idx], arr[par_idx]  # Swap

        Heapify(arr, n, par_idx)
   
def Build_Max_Heap(arr):
    n = len(arr)

    # Build max heap
    for idx in range(n, -1 ,-1):
        Heapify(arr, n, idx)

    # Swap(first, last) and extract 
    for idx in range(n-1, 0, -1):
        arr[0], arr[idx] = arr[idx], arr[0] 
        Heapify(arr, idx, 0)
    
    return arr





# ============================================================================================
arr1 = [18, 95, 34, 57, 23, 17, 13, 200]#, 61, 43, 22, 12, 3, 7, 8, 15, 32, 28, 24, 103, 100, 35]
#Sorted_Array  = Sort_Array(arr1)
#Sorted_Array  = Heapify(arr1, len(arr1) , 0)
Sorted_Array  = Build_Max_Heap(arr1)

print(arr1)