import numpy as np

def perceptron_train(X, Y, maxE=1000, eta=1.0, randState=0):
    rng = np.random.RandomState(randState)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0

    for i in range(maxE):
        err = 0
        for i in rng.permutation(n):
            xi = X[i]
            yi = Y[i]
            yHat = 1.0 if (np.dot(w, xi) + b) >= 0 else -1.0
            if yHat != yi:
                w += eta * yi * xi
                b += eta * yi
                err += 1

            if err == 0:
                break
    return w, b


def perceptron_test(X_test, Y_test, w, b):
    X_test = np.asarray(X_test, dtype=float)
    Y_test = np.asarray(Y_test, dtype=float)
    preds = np.where(X_test @ np.asarray(w, dtype=float) + float(b) >= 0, 1.0, -1.0)
    return float(np.mean(preds == Y_test))

