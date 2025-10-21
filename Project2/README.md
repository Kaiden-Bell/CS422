# CS 422 - Project 2

**Author:** Kaiden Bell

## Files:
    1. `perceptron.py`
    2. `gradient_descent.py`
    3. `linear_classifier.py`
    4. `test_script.py`
    5. `data_1.txt, data_2.txt`

## Overview

Project 2 highlights three key machine learning algorithms:

## **Perceptron**
    * Trains data labeled with {-1, +1}
    * Updates weight vectors and bias whenever it incorrectly classifies a label
    * Stops if there are no errors found in an epoch

## **Gradient Descent**
    * Minimizes a function using the update rule:
    \[
        x ← x - η∇f(x)
    \]
    * Stops when the gradient falls below a specific tolerance or after a maximum step limit occurs.

## **Linear Classifiers**
    * Trains a linear model using defined loss values
    * Performs graident descent across multiple epochs.
    * Predicts get the sign of wx + b

#  Functions used:

### **perceptron.py**

`perceptron_train(X, Y, maxE=1000, eta=1.0, randState = 0)`
    * Trains the model and returns weights and biases
`perceptron_test(X_test, Y_test, w, b)` 
    * Tests the perceptron and returns acc vals.

### **gradient_descent.py** 
`gradient_descent(gradF, xinit, eta, tol=1e-4, maxSteps=10000)`
    * Performs grad desc until it converges and returns the optimal vector.

### **linear_classifier.py**

`linear_train(X, Y, Dldw, Dldb, eta, epochs=200, randState=0)` 
    * Trains linear classifer using stochastic grad desc
`linear_test(X_test, Y_test, w, b)` 
    * Tests the classifier using sign predicts, returns acc.

