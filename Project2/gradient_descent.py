import numpy as np

def gradient_descent(gradF, xinit, eta, tol=1e-4, maxSteps=10000):
    x = np.asarray(xinit, dtype=float).copy()
    for l in range(maxSteps):
        g = np.asarray(gradF(x), dtype=float)
        if np.linalg.norm(g) < tol:
            break
        x = x - eta * g
    return x
