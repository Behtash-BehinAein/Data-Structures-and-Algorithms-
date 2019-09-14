import os
clear = lambda: os.system('cls')
clear()

# Inputs the initial value of the counters to the sorting function 
def Sort_Array(arr):
    return Quick_Sort(arr, 0, len(arr)-1)
# ================================================================

# Makes sure the counters do not overlap  
# Fetches the reference value 
# Calls the flipper function
# Recurses to itself for the partitioned portions  
def Quick_Sort(arr, cnt_L, cnt_R):

    # Base Case
    if cnt_L>= cnt_R:
        return 

    # Elements are flipped/not-flipped with respected to this reference value in the array
    Reference_Value = arr[int((cnt_L+cnt_R)/2)]

    # Call the flipper function and return the partition index
    Partition_idx = Flipper(arr, cnt_L, cnt_R, Reference_Value)

    # Recursion 
    Quick_Sort(arr, cnt_L, Partition_idx-1)
    Quick_Sort(arr, Partition_idx, cnt_R)

    return arr
# =================================================================


# Flips the elements when this is a valid operation to perform 
def Flipper(arr, cnt_L, cnt_R, Reference_Value):

    while cnt_L <= cnt_R:

        # Keep advancing the counters until the flipping elements are found on the left and right 
        while arr[cnt_L] < Reference_Value:
            cnt_L += 1
        while arr[cnt_R] > Reference_Value:
            cnt_R -= 1

        # Making sure the current counters have not overlapped
        if cnt_L <= cnt_R:

            # Perform the flip
            arr[cnt_L], arr[cnt_R]  = arr[cnt_R], arr[cnt_L] 

            # Advance the counters after the flip 
            cnt_L += 1
            cnt_R -= 1

    return cnt_L

# ============================================================================================
arr1 = [200, 13, 34, 57, 23, 17, 18, 95, 61, 43, 22, 12, 3, 7, 8, 15, 32, 28, 24, 103, 100, 35]
Sorted_Array  = Sort_Array(arr1)
print(Sorted_Array)