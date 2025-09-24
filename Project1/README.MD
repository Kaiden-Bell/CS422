# 1 Decision Trees

## DT_train_binary(X, Y, maxDepth):
    Implements a binary decision tree using info gain to select best feat at each split
    Entropy is used for loss
    Input feat matrix `X` is 2D -> `n_samples x n_features` with binary vals. `Y` is a 1D array of binary labels
    Each step:
        1. Compute entropy of curr labels
        2. Eval each feature's info gain
        3. Choose feature with highest info gain
    
    Recursively build left and right till:
        1. All labels in node are the same or no features are left, or max depth is reached.

    Tree is stored as a recursive dict with keys: "feature", "left", "right", and "prediction"

## DT_test_binary(X, Y, DT)

    Evals the acc of a trained decision tree
    For each row in `X`:
        1.) Traverses the tree by checking feat vals and following the branch till a leaf occurs
        2.) Returns the predicted label from leaf
    Prediciton are compared to `Y`
    Acc is computed as the fract of correct preds

## DT_make_prediction(x, DT)
    Predicts class label for one sample `x`
    Traveres the decision tree recursively:
        1.) If at a leaf node, return predict
        2.) Otherwise check the feat at this node and follow left or right
    Returns a scalar label (0 or 1)


# K-Nearest Neighbors
    KNN_test(X_train, Y_train, X_test, Y_test, K)
        Implements the K-Nearest Neighbors algo
        Both training and test sets can have real vals of any dim
        Each sample:
            1.) Compute Euclidean dist to all samples
            2.) Selects the indices of the K closest pts
            3.) Perform majority vote among their labels
        Collect preds for all test samples
        Return acc as the fract of predicted labels


# K-Means Clustering
    K_Means(X, K, mu)
        Implements the K-means clustering algo
        Input `X` is a set of a feature vectors
        Input `mu` is an array of inital cluster centers:
            1.) If `mu` is empty, randomly selct K points from `X` as inital centers.
            2.) Otherwise, use provided centers
        Update clusters till convergence:
            1.)  Assign each sample to the nearest cluster
            2.) Recompute each cluster center as the mean of all pts
            3.) Repeat until center stops changing
        Return clusters as a numpy array of shape.