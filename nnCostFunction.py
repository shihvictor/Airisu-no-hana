import numpy as np
from sigmoid import sigmoid
from math import isfinite

# Problem is log getting a value such as 0.999999 which leads to NaN Cost. This is caused by the trained nn_params being too large. 
# Fixed(?) by adding small value to value in log to avoid log(0).
def nnCostFunction(nn_params, X, y, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, lmbda):
    """
    Calculates the cost and the gradient of the NN.
    
    Returns: a 1x2 array containing the cost and the unraveled gradient matrix.
     """
    #print("cost fn running")
    m = X.shape[0]
    # FP
    Theta_1 = np.reshape(nn_params[0:HIDDEN_LAYER_SIZE*(INPUT_LAYER_SIZE+1)], (HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE+1), order = 'F').copy() # (layer2size x inputlayersize+1)
    Theta_2 = np.reshape(nn_params[HIDDEN_LAYER_SIZE*(INPUT_LAYER_SIZE+1):], (OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE+1), order = 'F').copy()
    # correct :D

    #row i of a or z corresponds to component (a or z) of NN using X(i).
    A_1 = np.hstack((np.ones((m, 1), dtype = int), X)) # (90x5). This is the input layer matrix = X with bias elements.
    Z_2 = A_1.dot(Theta_1.T) # dot = matrix multiplication 90x4
    A_2 = np.hstack((np.ones((m, 1), dtype = int), sigmoid(Z_2))) # (90x5). This is the second layer matrix = 4 activation elts plus a bias elt.
    Z_3 = A_2.dot(Theta_2.T) # 90x3
    A_3 = sigmoid(Z_3) # (90x3) Each row is the hypothesis layer for Ex(i)
    # These layer computations are correct :D

    """debugging"""
    if (0 in A_3) == True:
        print("\n=== DEBUG ===")
        print("\nTheta1 =" + str(Theta_1))
        print("Theta2 =" + str(Theta_2))
        print("X = " + str(X))
        print("y = " + str(y))
        print("inputlayersize = " + str(INPUT_LAYER_SIZE))
        print("hiddenlayersize = " + str(HIDDEN_LAYER_SIZE))
        print("outlayersize = " + str(OUTPUT_LAYER_SIZE))
        print("lmbda = " + str(lmbda))
        print("m = " + str(m))
        print("A_1 = " + str(A_1))
        print("Z_2 = " + str(Z_2))
        print("A_2 = " + str(A_2))
        print("Z_3 = " + str(Z_3))
        print("A_3 = " + str(A_3))
    """"""

    """============= Cost fn ============="""
    C = np.identity(OUTPUT_LAYER_SIZE, dtype=int)
    y_Logical = np.zeros((m, OUTPUT_LAYER_SIZE), dtype=int)
    for i in range(0, m):
        y_Logical[i, :] = C[int(y[i]-1), :]# class 0 = [1, 0, 0], class 1 = [0, 1, 0], etc...
    # y_Logical is correct :D

    # astype(float) changes the elts of A_3 from dtype('O') = objects TO float.
    """debugging"""
    tt1 = -1/m
    tt2 = np.log(A_3.astype(float))
    tt3 = np.multiply(y_Logical, np.log(A_3.astype(float)))
    tt4 = (1 - y_Logical)
    tt5 = np.log(1 - A_3.astype(float))
    tt6 = np.multiply((1 - y_Logical), np.log(1 - A_3.astype(float)))
    tt7 = np.sum(np.multiply(y_Logical, np.log(A_3.astype(float))) + np.multiply((1 - y_Logical), np.log(1 - A_3.astype(float))))
    """"""

    j_temp = -1/m * np.sum(np.multiply(y_Logical, np.log(A_3.astype(float))) + np.multiply((1 - y_Logical), np.log(1 - A_3.astype(float)+10**-10)))
    reg_Term = lmbda/(2*m) * (np.sum(np.power(Theta_1[:, 1:], 2)) + np.sum(np.power(Theta_2[:, 1:], 2))) #
    J = j_temp + reg_Term
    #print("J done")
    # debugging check
    if isfinite(j_temp) == False:
        print("\n")
        # print(nn_params)
        # print("\ntt1 = "+ str(tt1))
        # print("tt2 = "+ str(tt2))
        # print("tt3 = "+ str(tt3))
        # print("tt4 = "+ str(tt4))
        # print("tt5 = "+ str(tt5))
        # print("tt6 = "+ str(tt6))
        # print("tt7 = "+ str(tt7))

    """============= BP ============="""
    D_3 = A_3 - y_Logical #90x3
    D_2 = np.multiply(np.multiply(D_3.dot(Theta_2[:, 1:]), A_2[:, 1:]), (1 - A_2[:, 1:])) # ignores hidden layer bias elts bc no connection to input layer.
    Delta1 = np.dot(D_2.T, A_1) # same dim as respective Theta.
    Delta2 = np.dot(D_3.T, A_2)
    # Unregularized gradients for Theta matrices.
    Theta1_grad = 1/m * Delta1 + lmbda/m * np.hstack((np.zeros((Theta_1.shape[0], 1)), Theta_1[:, 1:]))
    Theta2_grad = 1/m * Delta2 + lmbda/m * np.hstack((np.zeros((Theta_2.shape[0], 1)), Theta_2[:, 1:]))
    #print("BP done")
    #print("J = " + str(J))
    #print("grad = \n" + str(np.ravel(np.vstack((Theta1_grad, Theta2_grad)))))
    # Unroll back.
    return [J, np.hstack((np.ravel(Theta1_grad.T), np.ravel(Theta2_grad.T)))]