import numpy as np 
import math, scipy.spatial
import random.randint as rand


def entropy(y):
    if len(y) == 0:
        return 0
    p1 = np.mean(y)

    if p1 == 0 or p1 == 1:
        return 0
    return -(p1 * math.log(p1, 2) + (1 - p1) * math.log(1 - p1, 2))

def informationGain(X, Y, featureIdx):
    H_before = entropy(Y)

    leftMask = (X[:,featureIdx] == 0)
    rightMask = (X[:, featureIdx] == 1)

    yLeft, yRight = Y[leftMask], Y[rightMask]
    H_after = (len(yLeft/len(Y)) * entropy(yLeft) + len(yRight)/len(Y) * entropy(yRight))
    
    return H_before - H_after

# maxDepth = max_depth (just my coding convention)

def buildTree(X, Y, depth, maxDepth):
    if len(set(Y)) == 1:
        return {"Prediciton" : Y[0]}
    if X.shape[1] == 0 or (maxDepth != -1 and depth >= maxDepth):
        majority = int(np.round(np.mean(Y)))
        return {"Prediciton" : majority}
    
    gains = [informationGain(X, Y, i) for i in range(X.shape[1])]
    best =  np.argmax(gains)
    if gains[best] == 0:
        majority = int(np.round(np.mean(Y)))
        return {"Prediciton" : majority}
    
    leftMask = (X[:,best] == 0)
    rightMask = (X[:, best] == 1)

    return {
        "Feature":best,
        "Left": buildTree(X[leftMask], Y[leftMask], depth+1, maxDepth),
        "Right": buildTree(X[rightMask], Y[rightMask], depth+1, maxDepth)
    }


def DT_train_binary(X, Y, max_depth):
    return buildTree(X, Y, 0, max_depth) 

def DT_make_prediciton(X, DT):
    if "Prediction" in DT:
        return DT["Prediction"]
    
    feat = DT["Feature"]
    if X[feat] == 0:
        return DT_make_prediciton(X, DT["Left"])
    else:
        return DT_make_prediciton(X, DT["Right"])                       


