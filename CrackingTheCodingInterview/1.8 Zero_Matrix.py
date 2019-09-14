import os
clear = lambda: os.system('cls')
clear()


# You solved this better than the book 
def ZeroMatrix(array):

    Nr = len(array)
    Nc = len(array[0])

    for ii in range(Nr):        
        for jj in range(Nc):
            if array[ii][jj] == 0:
                array[ii][0 ] = 0
                array[0 ][jj] = 0

    for ii in range(Nr):        
        for jj in range(Nc):
            if array[0][jj] == 0:
                array[ii][jj] = 0

            if array[ii][0] == 0:
                array[ii][jj] = 0
    return array



def PrintMatrix(array):
    for row in array:
        print(row)

#==================================================================================



array = [[1,2,3], [4,0,6] , [7,8,9]]
#array = [[1,2,3,4], [5,6,0,8], [9,10,11,12], [13,14,15,16]]


PrintMatrix(array)
print(' '*10)
PrintMatrix(ZeroMatrix(array))
print(' '*10)
