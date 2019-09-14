import os
clear = lambda: os.system('cls')
clear()

def Bubble_Sort(arr):
    cnt = 1
    while cnt<len(arr):   # This is just to repeat the swaps n times 
        for ee in range(len(arr)-1):

            if arr[ee] > arr[ee+1]:
                arr[ee], arr[ee+1]   = arr[ee+1], arr[ee]
        cnt +=1 
    return arr


# ============================================
arr1 = [200, 13, 34, 57, 23, 17, 18, 95, 61, 43, 22, 12, 3, 7, 8, 15, 32, 28, 24, 13, 100, 97]

Sorted_Array  = Bubble_Sort(arr1)
print(Sorted_Array)




