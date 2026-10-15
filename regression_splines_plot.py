from linalgsolve_remake import luke_linalgsolve

import numpy as np

import matplotlib as plt

def reg_plot(x, knots, y):
    """calculates a cubic regression spline, knot should be
    a subvector of x

    Parameters
    ----------
    x, knots, y : np arrays

    Returns
    -------
    No return, constructs a plot
    """

    #A needs a leading column of 1's to account for the matrix multiplcation down the line

    n = len(x) #returns size of x as a scalar

    K = len(knots) #returns size of knots array

    #building A

    a_col_0 = np.array(np.ones(n)) #shape (13,)

    a_col_1 = x #shape (13,)

    a_col_2 = x**2 #shape (13,)

    a_col_3 = x**3 #shape (13,)

    a_col_k = np.zeros((n, K)) #13xK

    a_col_k = a_col_k - knots #shape (13, K)

    a_col_k = a_col_k + x[:, None] #shape (13, K)

    a_col_k = np.maximum(0, a_col_k) #shape (13, K)

    a_col_k = a_col_k**3 #shape (13, K)

    A = np.column_stack((a_col_0, a_col_1, a_col_2, a_col_3, a_col_k)) #shape (13, K+4)

    a = luke_linalgsolve(A.T @ A, A.T @ y)

    t = np.linspace(x[0], x[-1], num = 1000) #shape of (1000,)

    #making b

    b_col_0 = np.ones(len(t)) #shape (1000,)

    b_col_1 = t #shape (1000,)

    b_col_2 = t**2 #shape (1000,)

    b_col_3 = t**3 #shape (1000,)

    b_col_k = np.zeros((len(t), K)) #shape (1000, K)

    b_col_k = b_col_k - knots #shape (1000, K) 

    b_col_k = b_col_k + t[:, None] #shape (1000, K)

    b_col_k = np.maximum(0, b_col_k) #shape (1000, K)

    b_col_k = b_col_k**3 #shape (1000, K) 

    B = np.column_stack((b_col_0, b_col_1, b_col_2, b_col_3, b_col_k)) #shape (1000, K+4)

    plt.scatter(x, y)

    plt.plot(t, B @ a)

    plt.show()