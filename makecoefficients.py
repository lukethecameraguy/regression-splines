from linalgsolve_remake import luke_linalgsolve

import numpy as np

def make_coefficients(x, y):
    """Returns the a, b, c, d vectors for the cubic splines with
    natural boundary conditions.

    Parameters
    ----------
    x, y numpy vectors of shape (n+1,)

    Returns
    -------
    4 numpy vectors a, b, c, d of shape (n,)
    """

    a = y[:-1] #n

    h = x[1:] - x[:-1] #shape n

    y_1 = y[1:] #shape n

    y_2 = y[:-1] #n

    ydif = y_1 - y_2 #n

    z = 3*(ydif[1:] / h[1:] - ydif[:-1] / h[:-1]) #shape n-1

    diagonal = 2*((h[:-1] + h[1:])) #shape n-1

    l = np.diag(h[1:-1], 1) #shape n-2
    l1 = np.diag(h[1:-1], -1) #shape n-2
    l2 = np.diag(diagonal, 0) #shape n-1 x n-1

    A = l + l1 + l2 #shape n-1 x n-1

    c = luke_linalgsolve(A, z) #shape n-1

    c_combined = np.concatenate(([0], c, [0])) #n+1

    c = c_combined[:-1] #shape n

    b_1 = ((y[1:]) - y[:-1]) / h #n

    b_2 = (h*(2*(c_combined[:-1])+c_combined[1:])) / 3 #n

    b = b_1 - b_2 #n

    d = ((c_combined[1:]) - c_combined[:-1]) / (3 * h) #n

    return a, b, c, d #shape n
