# Iris Flower Classification
The purpose of this program was to self review and apply certain concepts within the Machine Learning online course authorized by Stanford and offered through Coursera and, at the same time, develop an entire program from scratch to transition from Matlab to Python.

## Libraries Used
* Numpy, Pandas, Scipy, Matplotlib

## Project Description
Dataset was obtained from https://archive.ics.uci.edu/ml/datasets/Iris through https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

One difficulty was implementing the entire program from start to finish, unlike the implementation of only specific functions in Andrew Ng's Machine Learning Course assignments. This helped me understand the order in which I should implement each part of the NN program as well as how to tie together each algorithm within the NN. 

### Implementation Process
1. Set up architecture of NN
2. Dataset preparation
    * loading data
    * changing labels from string to integer representation
    * feature scaling
    * partition dataset into Training, Cross Validation, and Test set
3. Random initialize weights
4. NN algorithms:
    * forward propagation to get hypothesis
    * cost function of NN
    * back propagation to get gradients
5. NN gradient checking
6. Plot learning curves:
    * 1st plot shows error of training set and cv set with respect to the number of training examples
    * 2nd plot shows error of training set and cv set with respect to a set of lambda values
7. Training NN and predict on training set and test set
8. Perform Error Analysis using Error Metrics on both the training and test set:
    * Precision
    * Recall
    * F1-score
    * Classification accuracy 
  
