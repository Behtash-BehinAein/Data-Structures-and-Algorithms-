import os
clear = lambda: os.system('cls')
clear()


class Node():

    def __init__(self, value):
        self.left  = None
        self.right = None 
        self.value = value

#===========================


def printTreeInOrder(node):

    if node: 
        printTreeInOrder(node.left)
        print(node.value)
        printTreeInOrder(node.right)

#===========================

def printTreePreOrder(node):

    if node: 
        print(node.value)
        printTreePreOrder(node.left)
        printTreePreOrder(node.right)

#===========================

def printTreePostOrder(node):

    if node: 
        printTreePostOrder(node.left)
        printTreePostOrder(node.right)
        print(node.value)





#==================================================================================
root = Node(1) 
root.left      = Node(2) 
root.right     = Node(3) 
root.left.left  = Node(4) 
root.left.right  = Node(5) 
#==================================================================================


print('This is PreOrder tree traversal and print'), printTreePreOrder(root) 


#print('This is InOrder tree traversal and print'), printTreeInOrder(root) 


#print('This is PostOrder tree traversal and print'), printTreePostOrder(root) 

