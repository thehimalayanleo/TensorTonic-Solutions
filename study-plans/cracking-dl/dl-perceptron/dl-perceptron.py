import numpy as np

def perceptron(X, y, lr=0.1, epochs=100):
    """
    Returns: Tuple of (weights as list of floats, bias as float)
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    y = np.reshape(y, (y.shape[0], 1))
    w = np.zeros((X.shape[1], 1))
    b = 0.0
    n, p = X.shape[0], X.shape[1]
    for epoch in range(epochs):
        for indx in range(X.shape[0]):
            sample = np.reshape(X[indx, :], (1, p))
            pred = np.dot(sample, w) + b 
            yh = 1.0 if pred >= 0 else 0.0
            error = y[indx, 0] - yh
            w += lr * error * sample.T
            b += lr * error
        '''pred = np.dot(X, w) + b 
        yh = np.reshape(np.array([1 if z >= 0 else 0 for z in pred]), (len(pred), 1))
        error = y - yh
        w += lr * np.dot(X.T, error)
        b += lr * np.sum(error)'''

    return w.flatten(), b