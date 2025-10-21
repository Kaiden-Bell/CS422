import numpy as np

def linear_train(X, Y, dLdw, dLdb, eta, epochs=200, randState=0):
    rng = np.random.RandomState(randState)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0

    for l in range(epochs):
        for i in rng.permutation(n):
            gradW = dLdw(X[i], Y[i], w, b)
            gradB = dLdb(X[i], Y[i], w, b)
            w -= eta * gradW
            b -= eta * gradB
    return w, b

def linear_test(X_test, Y_test, w, b):
    X_test = np.asarray(X_test, dtype=float)
    Y_test = np.asarray(Y_test, dtype=float)
    preds = np.where(X_test @ np.asarray(w, dtype=float) + float(b) >= 0, 1.0, -1.0)
    return float(np.mean(preds == Y_test))

