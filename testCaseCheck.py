import numpy as np
from nnCostFunction import nnCostFunction
from checkNNGrads import checkNNGrads


def testCaseCheck():
    """Checking nnCostFunction implementation by using a test case from Coursera Stanford Machine Learning. https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/uPd5FJqnEeWWpRIGHRsuuw"""
    """
    Title: Test cases for ex4 nncostFunction()
    Author: Tom Mosher
    Date: Jan 27, 2020
    Code version: NA
    Availability: https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/uPd5FJqnEeWWpRIGHRsuuw
    """
    hl = 2
    il = 2
    lmbda = 4
    y = np.array([[4], [2], [3]])
    Xex = np.cos(np.array([[1, 2], [3, 4], [5, 6]]))
    nn = np.true_divide(np.arange(1, 19), 10)
    nl = 4
    result = nnCostFunction(nn, Xex, y, il, hl, nl, lmbda)
    print("J = " + str(result[0]))

    checkNNGrads(nn, Xex, y, il, hl, nl, lmbda)