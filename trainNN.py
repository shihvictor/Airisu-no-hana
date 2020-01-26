import numpy as np
from scipy.optimize import minimize
from nnCostFunction import nnCostFunction
from randInitializeWeights import randInitializeWeights

def trainNN(X, y, lmbda, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE):
    initial_Theta1 = randInitializeWeights(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE)
    initial_Theta2 = randInitializeWeights(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)
    nn_Params = np.hstack((np.ravel(initial_Theta1.T), np.ravel(initial_Theta2.T)))

    costfun = lambda nnP: nnCostFunction(nnP, X, y, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, lmbda)[0]
    gradfun = lambda nnP: nnCostFunction(nnP, X, y, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, lmbda)[1]

    result = minimize(costfun, nn_Params, method='CG', jac=gradfun, options={'disp': True, 'maxiter': 50})
    Theta_1 = np.reshape(result.x[0:HIDDEN_LAYER_SIZE*(INPUT_LAYER_SIZE+1)], (HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE+1), order = 'F') # (layer2size x inputlayersize+1)
    Theta_2 = np.reshape(result.x[HIDDEN_LAYER_SIZE*(INPUT_LAYER_SIZE+1):], (OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE+1), order = 'F')
    return np.hstack((np.ravel(Theta_1.T), np.ravel(Theta_2.T)))