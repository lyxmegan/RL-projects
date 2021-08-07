import numpy as np

if __name__ == '__main__':
    T = 5
    a = {
        0: np.array([0.5,-0.2]),
        1: np.array([0.2,0.1]),
        }

    s = {
        0: np.array([0,0]),
        1: np.array([1,-0.2]),
        2: np.array([1.2,-0.1]),
        5: np.array([3.2,1.3]),
        }

    G = 0
    gamma = 0.9
    R = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: -1
        }
    alpha = 0.1

    W = np.array([[0,0], [0,0]])

    for t in range(T):
        for k in range(t + 1, T+1):
            G = gamma**(k-t-1) * R[k]
        r = a[t] - W.T.dot(s[t])
        W = W + alpha * gamma**t * G * np.outer(r, s[t])
        print('W_{}: {}'.format(t, np.sum(W)))