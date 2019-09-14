import os
clear = lambda: os.system('cls')
clear()


def MergeSorted(arrL, arrR):

    n_L = len(arr1)
    n_R = len(arr2)

    arr = [0] * (n_L + n_R)

    cnt_L = 0
    cnt_R = 0
    nn = 0

    while cnt_L < n_L and cnt_R < n_R:

        if arrL[cnt_L] < arrR[cnt_R]:
            arr[nn] = arrL[cnt_L]
            cnt_L += 1 
        else:
            arr[nn] = arrR[cnt_R]
            cnt_R += 1
        nn += 1

    while cnt_L < n_L:
        arr[nn] = arrL[cnt_L]
        nn += 1
        cnt_L += 1 

    while cnt_R < n_R:
        arr[nn] = arrR[cnt_R]
        nn += 1
        cnt_R += 1   

    return arr



arr1 = [17, 18, 23, 34 , 57,  93, 95,  200]#
arr2 = [11, 36, 55, 71,  92, 103, 125, 130]
MergedArray  = MergeSorted(arr1, arr2)
print(MergedArray)