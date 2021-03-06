import numpy as np
from nnCostFunction import nnCostFunction
from trainNN import trainNN

def learningCurveLambda(X, y, X_CV, y_CV, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE):
    """Calculates the training set error and cross validation set error w.r.t a set of lambda values for use in plotting the learning curve to optimize lambda."""
    lmbda_values = np.array([[0], [0.001], [0.003], [0.01], [0.03], [0.1], [0.3], [1], [3], [10]])
    error_Train = np.zeros((lmbda_values.shape[0], 1))
    error_CV = np.zeros((lmbda_values.shape[0], 1))

    for i in range(0, lmbda_values.shape[0]):
        nn_params = trainNN(X, y, lmbda_values[i], INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)
        error_Train[i, :] = nnCostFunction(nn_params, X, y, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, 0)[0]
        error_CV[i, :] = nnCostFunction(nn_params, X_CV, y_CV, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, 0)[0]

    if np.isfinite(error_Train).any() == False:
        print("\n===========NAN VALUE(S) DETECTED============\n")
        print(nn_params)
    if np.isfinite(error_CV).any() == False:
        print("\n===========NAN VALUE(S) DETECTED============\n")
        print(nn_params)

    return lmbda_values, error_Train, error_CV