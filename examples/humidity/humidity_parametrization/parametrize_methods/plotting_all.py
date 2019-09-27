import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
import pickle

path = 'all_parametrize_twister/txt_files/'
filenames = ['error_[55,-144]_[58,-53]', 'error_[13,-144]_[15,-53]',
             'error_[32,-144]_[32,-53]']
x = np.array([np.zeros(8) for i in range(736)])
for filename in filenames:
    f = open(path + filename + '.txt','r')
    lines = f.readlines()
    # x = np.array([np.zeros(8) for i in range(len(lines))])
    for i in range(len(lines)):
        params = lines[i].split()[-8:]
        params = np.array([float(p) for p in params])
        x[i] = params
    f.close()

parameters = {'p0':x[:,0], 'p1':x[:,1], 'm0':x[:,2], 'm':x[:,4],
              'x':x[:,5], 'y':x[:,6], 'b':x[:,7]}

df = pd.DataFrame(parameters)
g = sns.pairplot(df, diag_kind="kde")
plt.show()

means = {}
keys = df.keys()
for key in keys:
    means[key] = df[key].mean()

print(means)
