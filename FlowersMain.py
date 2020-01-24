import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from randInitializeWeights import randInitializeWeights
from nnCostFunction import nnCostFunction
from checkNNGrads import checkNNGrads
from predict import predict
from learningCurve import learningCurve
from testCaseCheck import testCaseCheck
from learningCurveLambda import learningCurveLambda

print("RUNNING ML")
#Data is already featureScaled.
#headersForFeatsAndLabel = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
print("Loading dataset.")
"""--- Iris dataset ---"""
dataset = pd.read_csv('iris.csv', header = None)
"""--- Digit recog dataset ---"""
# data = scipy.io.loadmat('ex4data1.mat')
# dataset = np.hstack((data['X'], data['y']))
""""""
# print("Dataset Preview:\n", dataset.head())
# print("Dimensions: ", dataset.shape)

"""==================== Set up architecture.===================="""
INPUT_LAYER_SIZE = 4
HIDDEN_LAYER_SIZE = 4
OUTPUT_LAYER_SIZE = 3
NUMBER_OF_CLASSES = 3
# INPUT_LAYER_SIZE = 400
# HIDDEN_LAYER_SIZE = 25
# OUTPUT_LAYER_SIZE = 10
# NUMBER_OF_CLASSES = 10


# Parameters
initial_Theta1 = randInitializeWeights(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE)
initial_Theta2 = randInitializeWeights(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)
nn_Params = np.hstack((np.ravel(initial_Theta1.T), np.ravel(initial_Theta2.T))) # combining Theta matrices together into 1 vec.
# Put dataset in terms of X and y
"""--- For .csv ---"""
temp1 = dataset.to_numpy() # changes dataframe to numpy matrix for numpy matrix slicing.
"""--- For .mat ---"""
# temp1 = dataset
""""""
# shuffle order of examples.
np.random.shuffle(temp1)

X = temp1[:, 0:temp1.shape[1]-1]
y = temp1[:, temp1.shape[1]-1]
y = y[np.newaxis, :].T # newaxis increases current dimension by 1.
m = X.shape[0]
n = X.shape[1]

# change labels from string to integer representation.
"""For Iris dataset """
for i in range(0, m):
    if y[i] == "Iris-setosa":
        y[i] = 1
    elif y[i] == "Iris-versicolor":
        y[i] = 2
    elif y[i] == "Iris-virginica":
        y[i] = 3
X = X.astype(float)

"""================= Partitioning Dataset ================="""
# Partitioning data into Training, Cross Validation, and Test set.
X_Train = X[0:int(m*.6), :]
y_Train = y[0:int(m*.6), :] # These are float64. Need to cast labels to int.
X_CV = X[int(m*.6):int(m*.8), :]
y_CV = y[int(m*.6):int(m*.8), :]
X_Test = X[int(m*.8):, :]
y_Test = y[int(m*.8):, :]

y_Train = y_Train.astype(int)
y_CV = y_CV.astype(int)
y_Test = y_Test.astype(int)

#temp lambda val. Need to implement auto lambda selection. see ex5.
# lmbda = 3
# J_and_Grad = nnCostFunction(nn_Params, X_Train, y_Train, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, lmbda)
# J = J_and_Grad[0]
# Grad = J_and_Grad[1]

print("\n================== CHECKING NN GRADIENTS ==================")
# checkNNGrads(nn_Params, X_Train, y_Train, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, lmbda)

print("\n=============== Test case checking ===============")
testCaseCheck()

""" ================== PLOTTING LEARNING CURVES =================="""
lmbda = 0
m = 30
error_Train, error_CV = learningCurve(X_Train, y_Train, X_CV, y_CV, lmbda, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, m)
plt.figure()
plt.plot(range(m), error_Train, color='b', lw=0.5, label='Train')
plt.plot(range(m), error_CV, color='g', lw=0.5, label='Cross Validation')
plt.title('Learning Curve for NN')
plt.legend()
plt.xlabel('Number of training examples')
plt.ylabel('Error')

maxErrorValue = np.vstack((error_Train, error_CV)).max()

plt.xlim(0, m)
plt.ylim(0, maxErrorValue)
plt.legend(loc='upper right', shadow=True, fontsize='x-large', numpoints=1)
plt.show()

""" ================== PLOTTING LEARNING CURVES (for lambda) =================="""
lmbda_values, error_Train, error_CV = learningCurveLambda(X_Train, y_Train, X_CV, y_CV, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)
plt.figure()
plt.plot(lmbda_values, error_Train, color='b', lw=0.5, label='Train')
plt.plot(lmbda_values, error_CV, color='g', lw=0.5, label='Cross Validation')
plt.title('Learning Curve 2 for NN')
plt.legend()
plt.xlabel('Lambda value')
plt.ylabel('Error')

maxErrorValue = np.vstack((error_Train, error_CV)).max()

plt.xlim(0, 1)
plt.ylim(0, maxErrorValue)
plt.legend(loc='upper right', shadow=True, fontsize='x-large', numpoints=1)
plt.show()

"""=============== Train NN & Predict ==============="""
print("\n===============TRAINING NN===============")
lmbda = 2
costfun = lambda nnP: nnCostFunction(nnP, X_Train, y_Train, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, lmbda)[0]
gradfun = lambda nnP: nnCostFunction(nnP, X_Train, y_Train, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, lmbda)[1]

# res = minimize(fun = costfun, x0 = nn_Params, method='CG', jac = gradfun, options={'disp':True, 'maxiter':50})
result = minimize(costfun, nn_Params, method='CG', jac=gradfun, options={'disp': True, 'maxiter': 50})
Theta_1 = np.reshape(result.x[0:HIDDEN_LAYER_SIZE*(INPUT_LAYER_SIZE+1)], (HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE+1), order = 'F') # (layer2size x inputlayersize+1)
Theta_2 = np.reshape(result.x[HIDDEN_LAYER_SIZE*(INPUT_LAYER_SIZE+1):], (OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE+1), order = 'F')

Hyp = predict(Theta_1, Theta_2, X_Train)
HypResult = [Hyp == y_Train]
trainAcc = float(np.sum(np.array([Hyp == y_Train], dtype = int))) / float(y_Train.shape[0]) * 100
print("Training set accuracy : " + str(trainAcc))

Hyp = predict(Theta_1, Theta_2, X_Test)
test1 = [Hyp == y_Test]
testAcc = float(np.sum(np.array([Hyp == y_Test], dtype = int))) / float(y_Test.shape[0]) * 100
print("Test set accuracy : " + str(testAcc))




print("\n===============FINISHED===============")