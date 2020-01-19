import numpy as np
import pandas as pd
from randInitializeWeights import randInitializeWeights
from nnCostFunction import nnCostFunction
from checkNNGrads import checkNNGrads

print("\n===RUNNING ML===")
#Data is already featureScaled.
#headersForFeatsAndLabel = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
dataset = pd.read_csv('iris.csv', header = None)
print("Dataset Preview:\n", dataset.head())
print("Dimensions: ", dataset.shape)

# Set up architecture.
INPUT_LAYER_SIZE = 4
HIDDEN_LAYER_SIZE = 4
OUTPUT_LAYER_SIZE = 3
NUMBER_OF_CLASSES = 3

# Parameters
initial_Theta1 = randInitializeWeights(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE)
initial_Theta2 = randInitializeWeights(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)
# nn_Params = np.vstack((initial_Theta1, initial_Theta2)) # combining Theta matrices together into 1 var.
# nn_Params = np.ravel(nn_Params) # unraveled into vector.

nn_Params = np.hstack((np.ravel(initial_Theta1.T), np.ravel(initial_Theta2.T)))
# Put dataset in terms of X and y
temp1 = dataset.to_numpy() # changes dataframe to numpy matrix for numpy matrix slicing.
X = temp1[:, 0:4]
y = temp1[:, 4]
y = y[np.newaxis, :].T # newaxis increases current dimension by 1.

m = X.shape[0]
n = X.shape[1]

# change labels from string to integer representation. 
for i in range(0, m):
    if y[i] == "Iris-setosa":
        y[i] = 0
    elif y[i] == "Iris-versicolor":
        y[i] = 1
    elif y[i] == "Iris-virginica":
        y[i] = 2


X_Train = X[0:int(m*.6), :]
y_Train = y[0:int(m*.6), :]

#temp lambda val. Need to implement auto lambda selection. see ex5.
lmbda = 3
J_and_Grad = nnCostFunction(nn_Params, X_Train, y_Train, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, lmbda)
J = J_and_Grad[0]
Grad = J_and_Grad[1]

print("=== CHECKING NN GRADIENTS ===")
checkNNGrads(nn_Params, X_Train, y_Train, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, lmbda)



print("=== Test case checking ===")

hl = 2
il = 2
lmbda = 4
y = np.array([[3], [1], [2]])
Xex = np.cos(np.array([[1, 2], [3, 4], [5, 6]]))
nn = np.true_divide(np.arange(1, 19), 10)
nl = 4
result = nnCostFunction(nn, Xex, y, il, hl, nl, lmbda)
print("J = " + str(result[0]))
print("grad = \n" + str(result[1]))

print("=== 2nd check ===")
checkNNGrads(nn, Xex, y, il, hl, nl, lmbda)

"""============ Train NN & Predict ============"""





print("===FINISHED===")