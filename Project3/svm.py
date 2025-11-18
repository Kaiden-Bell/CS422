import numpy as np
import matplotlib.pyplot as plt

"""
SINGLE CLASS

"""

def distance_point_to_huperplace(pt, w, b):
    pt = np.asarray(pt, dtype=float)
    w = np.asarray(w, dtype=float)

    magW = np.linalg.norm(w)
    if magW == 0:
        raise ValueError("w must be non-zero in the function")
    
    num = np.abs(np.dot(w, pt) + b)
    return num / magW


def svm_test_brute(w, b, x):
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)

    sc = np.dot(w, x) + b
    return 1 if sc >= 0 else -1


def compute_margin(data, w, b):
    w = np.asarray(w, dtype=float)
    magW = np.linalg.norm(w)

    if magW == 0:
        return 0.0
    
    margins = []

    for row in data:
        x = row[:2]
        y = row[2]

        signed = y * (np.dot(w, x) + b) / magW

        if signed <= 0:
            return 0.0
        
        margins.append(signed)
    
    return min(margins) if margins else 0.0


def plot_data_and_boundary(data, w, b):
    w = np.asarray(w, dtype=float)

    for row in data:
        x1, x2, y = row
        if y == 1:
            plt.plot(x1, x2, 'b+')
        else:
            plt.plot(x1, x2, 'ro')
    
    m = max(data[:, :2].max(), abs(data[:, :2].min())) + 1 
    xs = np.linspace(-m, m, 100)

    if np.abs(w[1]) > 1e-8:
        ys = (-w[0] * xs - b) / w[1]
        plt.plot(xs, ys, 'k-')
    else:
        if np.abs(w[0]) < 1e-8:
            raise ValueError("Cannot plot boundary! Both compnents of w are ~0")
        xVert = -b / w[0]
        plt.axvline(x=xVert, linestyle='-', color='k')

    plt.axis([-m, m, -m, m])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Training data + decision boundary')
    plt.show()


def svm_train_brute(training_data):
    data = np.asarray(training_data, dtype=float)
    N = data.shape[0]

    bestMarg = 0.0
    bestW = None
    bestB = None
    bestS = None

    for i in range(N):
        for j in range(i + 1, N):
            xi, yi = data[i, :2], data[i, 2]
            xj, yj = data[j, :2], data[j, 2]

            if yi == yj: continue
                
            if yi == 1 and yj == -1:
                xpos, xneg = xi, xj
            elif yi == -1 and yj == 1:
                xpos, xneg = xj, xi
            else: continue


            w = xpos - xneg
            mid = (xpos + xneg) / 2

            if np.linalg.norm(w) < 1e-10: continue

            b = -np.dot(w, mid)
            margin = compute_margin(data, w, b)

            if margin > bestMarg:
                bestMarg = margin
                bestW = w
                bestB = b
                bestS = np.vstack([np.append(xpos, 1.0), np.append(xneg, -1.0)])
    
    if bestW is None:
        raise RuntimeError("Unable to seperate data via brute force")
    
    return bestW, bestB, bestS

"""
MULTI-CLASS

"""

def svm_train_multiclass(training_data):
    data = np.asarray(training_data, dtype=float)
    labels = data[:, 2].astype(int)

    y = int(labels.max())

    w = np.zeros((y, 2))
    b = np.zeros(y)

    for c in range(1, y+1):
        binaryDat = data.copy()
        binaryDat[:, 2] = np.where(labels == c, 1.0, -1.0)

        wC, bC, sC = svm_train_brute(binaryDat)

        w[c - 1, :] = wC
        b[c - 1] = bC
    
    return w, b

def svm_test_multiclass(W, B, x):
    W = np.asarray(W, dtype=float)
    B = np.asarray(B, dtype=float)
    x = np.asarray(x, dtype=float)

    scores = W @ x + B

    posInd = np.where(scores >= 0)[0]

    if posInd.size == 0:
        return -1
    
    bestClass = None
    bestDist = -np.inf

    for i in posInd:
        wi = W[i]
        magW = np.linalg.norm(wi)
        if magW < 1e-10: continue

        dist = scores[i] / magW
        if dist > bestDist:
            bestDist = dist
            bestClass = i + 1

    if bestClass is None: return -1

    return bestClass


def plot_data_and_boundaries(data, W, B):
    data = np.asarray(data, dtype=float)
    labels = data[:, 2].astype(int)

    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    markers = ['+', 'o', '*', 'x', 's', 'd', '^']

    for row in data:
        x1, x2, y = row
        index = int(y) - 1
        c = colors[index % len(colors)]
        m = markers[index % len(markers)]
        plt.plot(x1, x2, c + m)

    m = max(data[:, :2].max(), abs(data[:, :2].min())) + 1
    xS = np.linspace(-m, m, 200)
    numClasses = W.shape[0]
    
    for i in range(numClasses):
        w = W[i]
        b = B[i]
        magW = np.linalg.norm(w)

        if magW < 1e-10: continue

        if abs(w[1]) > 1e-8:
            yS = (-w[0] * xS - b) / w[1]
            plt.plot(xS, yS, c + '-')
        else:
            if abs(w[0]) > 1e-8:
                xVert = -b / w[0]
                plt.axvline(x=xVert, color=c)
    
    plt.axis([-m, m, -m, m])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Multi-class data + decision boundaries')
    plt.show()

"""
Main class for Single Class test

if __name__ == "__main__":
    from helpers import generate_training_data_binary

    data = generate_training_data_binary(3)
    w, b, S = svm_train_brute(data)
    print("w:", w)
    print("b:", b)
    print("Support vectors:\n", S)
    print("Margin:", compute_margin(data, w, b))
    plot_data_and_boundary(data, w, b)

"""

"""

Main class for multi class classification test

if __name__ == "__main__":
    from helpers import generate_training_data_multi

    data, Y = generate_training_data_multi(1)
    W, B = svm_train_multiclass(data)
    print("W:\n", W)
    print("B:\n", B)

    plot_data_and_boundaries(data, W, B)

    test_points = [
        np.array([-4, 0]), 
        np.array([0, 4]),
        np.array([4, 0]),
        np.array([0, -4]),
    ]

    for pt in test_points:
        y_pred = svm_test_multiclass(W, B, pt)
        print(f"x={pt}, predicted class={y_pred}")

"""






