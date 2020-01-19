import math


def sigmoid(Z):
    t1 = 1/(1+math.e**-Z)
    return t1