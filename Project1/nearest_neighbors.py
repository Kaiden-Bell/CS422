import numpy as np
from scipy.spatial import distance

def KNN_predict(x, X_train, Y_train, K):
    dist = [distance.euclidean(x, xT) for xT in X_train]

    neighborsIdx = np.argsort(dist)[:K]
    neighborLabel = Y_train[neighborsIdx]

    prediction = int(np.round(np.mean(neighborLabel)))
    return prediction 

def KNN_test(X_train, Y_train, X_test, Y_test, K):
    preds = [KNN_predict(x, X_train, Y_train, K) for x in X_test]
    preds = np.array(preds)

    return np.mean(preds == Y_test)
