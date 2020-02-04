
Not lin reg.
Maybe try NN or LogReg Multiclass.

Overall NN process

Pick network architecture

Train NN:
    rand init wts
    implem forward prop to get vector of hypothesis for each example in input.
    implem code to compute cost function J of NN.
    implem back prop to comput gradient terms for the weights.

Gradient checking: DONE

Minimize cost fn J using Conjugate Gradient algorithm.

Error Analysis:
    Learning Curves
    F1-score = 2 * (precision * recall) / (precision + recall)
        https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1

