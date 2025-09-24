import numpy as np 
import math

# Entropy

def entropy(y):
    if len(y) == 0:
        return 0
    p1 = np.mean(y)

    if p1 == 0 or p1 == 1:
        return 0
    return -(p1 * math.log(p1, 2) + (1 - p1) * math.log(1 - p1, 2))

# Binary

def informationGain(X, Y, featureIdx):
    H_before = entropy(Y)

    leftMask = (X[:, featureIdx] == 0)
    rightMask = (X[:, featureIdx] == 1)

    yLeft, yRight = Y[leftMask], Y[rightMask]
    H_after = (len(yLeft/len(Y)) * entropy(yLeft) + len(yRight)/len(Y) * entropy(yRight))
    
    return H_before - H_after

def buildTreeBinary(X, Y, depth, maxDepth):
    if len(set(Y)) == 1:
        return {"Prediction" : Y[0]}
    if X.shape[1] == 0 or (maxDepth != -1 and depth >= maxDepth):
        majority = int(np.round(np.mean(Y)))
        return {"Prediction" : majority}
    
    gains = [informationGain(X, Y, i) for i in range(X.shape[1])]
    best =  np.argmax(gains)
    if gains[best] == 0:
        majority = int(np.round(np.mean(Y)))
        return {"Prediction" : majority}
    
    leftMask = (X[:,best] == 0)
    rightMask = (X[:, best] == 1)

    return {
        "Feature":best,
        "Left": buildTreeBinary(X[leftMask], Y[leftMask], depth+1, maxDepth),
        "Right": buildTreeBinary(X[rightMask], Y[rightMask], depth+1, maxDepth)
    }

def DT_train_binary(X, Y, maxDepth):
    return buildTreeBinary(X, Y, 0, maxDepth) 

def DT_make_prediction(X, DT):
    if "Prediction" in DT:
        return DT["Prediction"]
    
    feat = DT["Feature"]
    if X[feat] == 0:
        return DT_make_prediction(X, DT["Left"])
    else:
        return DT_make_prediction(X, DT["Right"])                       


def DT_test_binary(X, Y, DT):
    preds = [DT_make_prediction(x, DT) for x in X]
    return np.mean(preds == Y)


