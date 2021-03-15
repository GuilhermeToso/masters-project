""" Big O of SCPL """


import numpy as np
import matplotlib.pyplot as plt
n = np.arange(0,30000)
def big_o(n, alpha):

    #a = (1 - alpha**2)*(n**2)
    f = ((1 - alpha**2)*n**2 + n*(alpha - 1))/2
    return f
alpha = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1])
cols = ['b','r','g','orange','k','c','m','gold','dimgray','chocolate']
for i in range(alpha.size):
    plt.plot(n,big_o(n,alpha[i]),cols[i], label =r'$\alpha = {0}$'.format(alpha[i]))
plt.plot(n,n**2)
plt.xlabel("N", fontsize=34, labelpad=30)
plt.ylabel(r"Operações", fontsize=30, labelpad=50)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid()
plt.legend(fontsize=20)
plt.show()
