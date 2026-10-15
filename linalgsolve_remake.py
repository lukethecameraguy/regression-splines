import numpy as np

def luke_linalgsolve(A, b):

    """
    Parameters
    ----------
    A numpy array nxn

    b numpy length (n) vector

    Returns
    -------
    Returns x, the length (n) solution vector.
    """

    #make copies of A and b so nothing gets destroyed

    A = A.copy().astype(float)

    b = b.copy().astype(float)

    #establish variables

    n = len(b)

    #the math shit

    for k in range(n-1):

        pivot_row = k + np.argmax(abs(A[k:, k])) #to determine row with largest entry

        if pivot_row != k: #swaps row with largest entry to top if not on top
            A[[k, pivot_row]] = A[[pivot_row, k]]
            b[[k, pivot_row]] = b[[pivot_row, k]]

        if A[k, k] == 0: #account for divide by zero so no errors
            raise ValueError("Divide by zero error. Matrix cannot be solved (is singular)")
        
        for i in range(k+1, n):
            factor = A[i, k] / A[k,k] #gets the factor to eliminate entries below pivot

            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]


    #the back substitution step

    x = np.zeros(n)
    
    for i in range(n-1, -1, -1): #calc x[i] by subtracting dot product of known var from b[i]

        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]


    return x

