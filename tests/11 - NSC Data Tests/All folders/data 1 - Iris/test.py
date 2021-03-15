import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

x = np.linspace(0,500,num=100)
a = np.linspace(0,0.05,num=100)
def sigmoid(x,a):
    return 1 - np.exp(-a*x)

# cols = cm.get_cmap('viridis',a.size).colors

# for i in range(a.size):
#     f = sigmoid(a[i],x)
#     plt.plot(x,f,c=cols[i])
# plt.show()

print(1 - np.exp(-np.linspace(0,20,num=50)))