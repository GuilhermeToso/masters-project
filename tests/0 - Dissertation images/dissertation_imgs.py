import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

x = np.linspace(0,5,num=101)

fig = plt.figure(num=1, dpi=100,figsize=(10,10))
ax = fig.add_subplot(121)
a1 = fig.add_subplot(122)

ax.scatter(x[6:27], np.random.normal(1,0.5,x[6:27].size), s=50, c='magenta')
ax.scatter(x[56:77], np.random.normal(1.5,0.5,x[56:77].size), s=50, c='red')
ax.scatter(x[6:27], np.random.normal(3.5,0.5,x[6:27].size), s=50, c='blue')
ax.scatter(x[66:87], np.random.normal(4,0.5,x[66:87].size), s=50, c='green')
ax.tick_params(axis='both', which='major', labelsize=25)
ax.set_xlabel(r"$A_{1}$", fontsize=25, labelpad=20)
ax.set_ylabel(r"$A_{2}$", fontsize=25, labelpad=20, rotation=0)
ax.set_xlim(0,5)
ax.set_ylim(0,5)


a1.scatter(x[35:70], np.random.normal(4.5,0.2,x[35:70].size), s=50, c='blue')
a1.scatter(x[15:85], 3.7 + 0.2*np.sin(3*x[15:85]) + np.random.normal(0,0.08,x[15:85].size), s=50, c='red')
a1.scatter(x[5:75],  3.3 + 0.25*np.sin(2*x[5:75]) - 0.8*x[5:75] + np.random.normal(0,0.2,x[5:75].size), s=50, c='green')
a1.scatter(x[55:90], np.random.normal(2,0.3,x[55:90].size), s=50, c='magenta')
a1.tick_params(axis='both', which='major', labelsize=25)
a1.set_xlabel(r"$A_{1}$", fontsize=25, labelpad=20)
a1.set_ylabel(r"$A_{2}$", fontsize=25, labelpad=20, rotation=0)
a1.set_xlim(0,5)
a1.set_ylim(0,5)

plt.show()

