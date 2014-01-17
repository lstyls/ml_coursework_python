import numpy as np

def weighted_least_squares(X, y, w):
    X = np.matrix(X)
    y = np.matrix(y)
    if len(y) == 1:
        y = y.T
        
    W = np.zeros((len(w),len(w)))
    for i in range(len(w)):
        W[i,i] = w[i]
    
    numerator = (X.T)*W*y
    denom = (X.T)*W*y
    beta = (denom.I)*numerator
    return beta
    