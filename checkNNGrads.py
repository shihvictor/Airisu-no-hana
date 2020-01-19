import numpy as np
from nnCostFunction import nnCostFunction

def checkNNGrads(nn_Params, X_Train, y_Train, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, lmbda) :
    
    
    numGrad = np.zeros((nn_Params.shape[0], 1))
    X_TrainForNum = X_Train[0:9, :]
    y_TrainForNum = y_Train[0:9, :]
    eps = 10**(-4)
    nn_Params_1 = np.copy(nn_Params)
    nn_Params_2 = np.copy(nn_Params)
    J_and_Grad = nnCostFunction(nn_Params, X_Train, y_Train, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, lmbda)
    grad = J_and_Grad[1].reshape((nn_Params.shape[0], 1))
    for i in range(0, nn_Params.shape[0]):
        nn_Params_1[i] = nn_Params_1[i] + eps
        nn_Params_2[i] = nn_Params_2[i] - eps
        J_and_Grad_1 = nnCostFunction(nn_Params_1, X_Train, y_Train, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, lmbda)
        J_and_Grad_2 = nnCostFunction(nn_Params_2, X_Train, y_Train, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, lmbda)
        numGrad[i] = (J_and_Grad_1[0] - J_and_Grad_2[0]) / (2*eps)
        nn_Params_1 = np.copy(nn_Params)
        nn_Params_2 = np.copy(nn_Params)

    print("Left: Analytical gradient, Right: Numerical gradient")
    print(np.hstack((grad, numGrad)))
    diff = np.linalg.norm(numGrad-grad)/np.linalg.norm(numGrad+grad);
    print("diff = " + str(diff))