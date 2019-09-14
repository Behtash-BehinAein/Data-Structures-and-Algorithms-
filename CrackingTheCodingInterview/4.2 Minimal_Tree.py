import os
clear = lambda: os.system('cls')
clear()


class TreeNode():

    def __init__(self, value):
        self.value = value
        self.left  = None
        self.right = None 


def SortedArrayToBST(arr):

    if not arr:  # Executes if we have an empty array
        return None 

    mid_idx = len(arr)//2

    Node       = TreeNode(arr[mid_idx])
    Node.left  = SortedArrayToBST(arr[:mid_idx])
    Node.right = SortedArrayToBST(arr[mid_idx+1:])


    return Node



def PrintTreeInOrder(Node):
    if Node:
        PrintTreeInOrder(Node.left)
        print(Node.value)
        PrintTreeInOrder(Node.right)



#==================================================================================

Sorted_Array = [2,4,6,8,10,20]
root = SortedArrayToBST(Sorted_Array)
#print(root.value)



print('This is InOrder tree traversal and print')
PrintTreeInOrder(root) 
