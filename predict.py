import numpy as np
from sigmoid import sigmoid

def predict(Theta_1, Theta_2, X):
    m = X.shape[0]
    A_1 = np.hstack((np.ones((m, 1)), X))
    Z_2 = A_1.dot(Theta_1.T)
    A_2 = np.hstack((np.ones((m, 1)), sigmoid(Z_2)))
    Z_3 = A_2.dot(Theta_2.T)
    A_3 = sigmoid(Z_3)
    result = np.argmax(A_3, axis = 1) +1
    return result[np.newaxis, :].T