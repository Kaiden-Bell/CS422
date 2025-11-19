# CS 422 - Project 3

**Author:** Kaiden Bell

## Files:
    1. `helpers.py`
    2. `svm.py`

## Overview

This project implements a brute-force SVM for binary and multi-class classification in 2D.
Instead of using libraries, the max margin sep is found by checking paris of points selecting the separator with max margin.

## Part 1 - Binary SVM

`distance_point_to_hyperplane(pt, w, b)`
    
Which computes:
    (|w ⋅ pt + b|) / ||w||
This formula is used to measure margin distances.

`compute_margin(data, w, b)`
For each point(x_i, y_i), comput ethe signed dist:
γ_i = y_i(w ⋅ pt + b) / ||w||
* If any γ_i ≤ 0, the separator misclassifies a point where margin = 0
* Otherwise margin = min over all γ_i
This ensures only valid separators are kept

`svm_test_brute(w, b, x)`
Uses a sign test where we determine label:
* +1 if w ⋅ pt + b ≥ 0
* -1 otherwise

`plot_data_and_boundary(data, w, b)`
Plots the data:
* +1 points are blue +
* -1 points as red o
* Decision boundary w ⋅ pt + b = 0
    * As a line if w_2 != 0
    * As a vert line if w_2 = 0

`svm_train_brute(training_data)` 
Core function for the project:
    1. Loop over all pairs of pts with opposite labels
    2. For each pair (+x, x-):
        * Construct a norm vect -> w = +x - x-
        * Take mid pt -> mid = (+x + x-) / 2
        * b = -w ⋅ m
        * Compute margin using compute_margin

    3. Select the sep (w,b) with max margin
    4. Return
        * w -> weight vec
        * b -> bias
        * sC -> the supp vec that gen this sep

## Part 2 - Multi-class SVM


`svm_train_multiclass(training_data)`
For each class c = 1..Y
    * Relabel training dat
        * Class c -> +1
        * All other classes -> -1
    * Train binary SVM using svm_train_brute
    * Store res wC, bC

Returns:
W -> Matrix of shape (Y,2)
B -> Vec shape (Y,)

Where:
* Row i of W -> to i + 1
* Elem i of B -> to i + 1

`svm_test_multiclass(W, B, x)`
For a test point:
    1. Comp scores:
         s_i = w_i ⋅ x + b_i
    2. Keep all classes with s_i >= 0
    3. If:
        * 0 classes -> return none
        * 1 class -> return that class
        * > 1 class -> choose the one where pt is farthest from boundary
            s_i / ||w_i||

`plot_data_and_boundaries(data, W, B)`
* Plots each class with unique color/marker
* Plots all Y decision boundaries
* Uses vert lines for w_2 = 0
Helps visualize how one-vs-rest divides space