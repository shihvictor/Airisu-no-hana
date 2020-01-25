import numpy as np
from nnCostFunction import nnCostFunction
from trainNN import trainNN

def learningCurve(X, y, X_CV, y_CV, lmbda, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, m):
    #m = X.shape[0]
    error_Train = np.zeros((m, 1))
    error_CV = np.zeros((m, 1))

    for i in range(1, m+1):
        nn_params = trainNN(X[0:i, :], y[0:i, :], lmbda, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)
        error_Train[i-1, :] = nnCostFunction(nn_params, X[0:i, :], y[0:i, :], INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, 0)[0]
        error_CV[i-1, :] = nnCostFunction(nn_params, X_CV, y_CV, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, 0)[0]

    if np.isfinite(error_Train).any() == False:
        print(nn_params)
    if np.isfinite(error_CV).any() == False:
        print(nn_params)

    return error_Train, error_CV

