import numpy as np

def f(D):
    return 1. / (1 + np.power(10, D/400))

def log_likelihood(elos, eloAdvantage, eloDraw, winTable, drawTable):
    l = 0
    for i in range(len(elos)):
        for j in range(len(elos)):
            l += winTable[i][j] * np.log(f(elos[i] - elos[j] - eloAdvantage + eloDraw))
            + 0.5 * drawTable[i][j] * np.log(1 - f(elos[i] - elos[j] - eloAdvantage + eloDraw) - f(elos[j] - elos[i] + eloAdvantage + eloDraw))
    return l

def get_log_likelihood_for_scipy(winTable, drawTable):
    def l(x):
        elos = x[0:-2]
        eloAdvantage = x[-2]
        eloDraw = x[-1]
        return log_likelihood(elos, eloAdvantage, eloDraw, winTable, drawTable)
    return l