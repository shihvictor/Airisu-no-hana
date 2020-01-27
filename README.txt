https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

Not lin reg.
Maybe try NN or LogReg Multiclass.

Overall NN process

pick network architecture

Train NN:
    rand init wts
    implem forward prop to get h(x(i)) for any input vector x(i).
    implem code to compute cost fn J.
    implem back prop to comput gradient terms.

    perform forward propagation and then back propagation using example x(i), y(i)
    get gradient matrix for layers 2

Gradient checking: DONE
Error Analysis:
    Learning Curves
    F1-score = 2 * (precision * recall) / (precision + recall)
        https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1