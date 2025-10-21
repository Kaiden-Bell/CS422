import numpy as np
from scipy.spatial import distance
from random import randint

def K_Means(X, K, mu):
    n_samples, n_features = X.shape

    if len(mu) == 0:
        ind = [randint(0, n_samples-1) for i in range(K)]
        mu = X[ind, :]
    else:
        mu = np.array(mu).reshape(K, n_features)
    
    while True:
        assign = []
        for x in X:
            dist = [distance.euclidean(x, c) for c in mu]
            assign.append(np.argmin(dist))
        assign = np.array(assign)

        new_mu = []
        for k in range(K):
            clust = X[assign == k]
            if len(clust) > 0:
                new_mu.append(np.mean(clust, axis=0))
            else:
                new_mu.append(mu[k])
        new_mu = np.array(new_mu)


        if np.allclose(mu, new_mu):
            break
        mu = new_mu
    return mu