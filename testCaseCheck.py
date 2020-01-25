import numpy as np
from nnCostFunction import nnCostFunction
from checkNNGrads import checkNNGrads

def testCaseCheck():
    hl = 2
    il = 2
    lmbda = 4
    y = np.array([[3], [1], [2]])
    Xex = np.cos(np.array([[1, 2], [3, 4], [5, 6]]))
    nn = np.true_divide(np.arange(1, 19), 10)
    nl = 4
    result = nnCostFunction(nn, Xex, y, il, hl, nl, lmbda)
    print("J = " + str(result[0]))

    checkNNGrads(nn, Xex, y, il, hl, nl, lmbda)