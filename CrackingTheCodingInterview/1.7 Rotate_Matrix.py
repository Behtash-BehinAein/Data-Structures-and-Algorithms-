 import os
clear = lambda: os.system('cls')
clear()


def Rotate_MatrixClk(array):

    N = len(array)

    for Layer in range(N-2):
        Last  = N-Layer-1
        for cc in range(Layer, Last):

            # Save the right wing
            Right = array[cc][Last]

            # Top --> Right  
            #   Right         Top
            array[cc][Last] = array[Layer][cc]  
              
            # Left --> Top
            #   Top            Left    
            array[Layer][cc] = array[N - (cc+1)][Layer]
   
            # Bottom --> Left
            #   Left                   Bottom 
            array[N - (cc+1)][Layer] = array[Last][N - (cc+1)]

            # Right --> Bottom
            #    Bottom               Right   
            array[Last][N - (cc+1)] = Right

    return array

def RotateMatrixAntiClk(array):
    N = len(array)
    for Layer in range((N-2)):
        Last = N - Layer - 1
        for idx in range(Layer, Last):
            
            # Save Left
            Left = array[idx][Layer]
            
            # Top --> Left
            array[idx][Layer] = array[Layer][N - (idx+1)]
            
            # Right --> Top
            array[Layer][N-(idx+1)] = array[N-(idx+1)][Last]
            
            
            # Bottom --> Right 
            array[N- (idx+1)][Last] = array[Last][idx]
            
            # Left --> Bottom
            array[Last][idx] = Left
    
    return array


def PrintMatrix(array):
    for row in array:
        print(row)

#==================================================================================



#array = [[1,2,3], [4,5,6] , [7,8,9]]
array = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]


PrintMatrix(array)
print(' '*10)
PrintMatrix(Rotate_MatrixClk(array))
print(' '*10)
PrintMatrix(RotateMatrixAntiClk(array))
