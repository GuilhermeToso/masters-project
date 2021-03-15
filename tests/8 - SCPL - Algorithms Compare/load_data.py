import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


data = np.load('data\\scpl_scores\\Scores.npy')
data_bcw = np.load('data\\scpl_scores\\Scores_bcw.npy')
print(data)

models = ['SCPL','LP', 'LS', 'SST+KNN', 'SST+NB', 'SST+DT','SST+SVM', 'TT+KNN', 'TT+NB', 'TT+DT','TT+SVM']

means = pd.DataFrame(data.mean(axis=1), columns=models, index=np.arange(5,20))
stds = pd.DataFrame(data.std(axis=1), columns=models, index=np.arange(5,20))

means.iloc[:,1:] = means.iloc[:,1:]*100

means_bcw = pd.DataFrame(data_bcw.mean(axis=1), columns=models, index=np.arange(1,15)*5)
stds_bcw = pd.DataFrame(data_bcw.std(axis=1), columns=models, index=np.arange(1,15)*5)

means_bcw.iloc[:,1:] = means_bcw.iloc[:,1:]*100


sns.heatmap(means_bcw, vmin=30, vmax=100, annot=True)
plt.show()

