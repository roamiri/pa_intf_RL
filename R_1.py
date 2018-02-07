import numpy as np

def R_1(p1, p2, g1, g2, beta, sigma2, gamma):

    gamma1 = (g1*p1)/(g1*p2*beta+sigma2)
    gamma2 = (g2*p2) / (g2*p1*beta + sigma2)
    reward = np.log2(1+gamma1/gamma) + np.log2(1+gamma2/gamma)
    return reward