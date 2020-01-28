import numpy as np
from sklearn import metrics

def f1score(predicted, actual):
    """Calculates the F1-score for Error Metrics."""
    m = actual.shape[0]

    PN_Matrix = np.zeros((3, 3)) # [i, j] corresponds to class i's tp, fn, fp
    for i in range(1, 4):
        tempA = (actual==i).astype(int)
        tempP = (predicted==i).astype(int)
        for j in range(0, m):
            a = tempA[j]
            p = tempP[j]
            if a == 1:
                if p == 1:
                    PN_Matrix[i-1, 0] = PN_Matrix[i-1, 0] + 1 # True pos.
                else:
                    PN_Matrix[i-1, 1] = PN_Matrix[i-1, 1] + 1 # False neg.
            else:
                if p == 1:
                    PN_Matrix[i-1, 2] = PN_Matrix[i-1, 2] + 1 # False pos.

    precision = PN_Matrix[:, 0] / (PN_Matrix[:, 0] + PN_Matrix[:, 2])
    recall = PN_Matrix[:, 0] / (PN_Matrix[:, 0] + PN_Matrix[:, 1])
    f1 = 2*precision*recall/(precision+recall) # This checks my own implementation of f1-score.
    # print("f1: " + str(f1)) 
    print(metrics.classification_report(actual, predicted, digits = 3))