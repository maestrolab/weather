import matplotlib.pyplot as plt
from scipy import interpolate
import seaborn as sns
import numpy as np
import pickle
import pandas as pd

# From path data
designs = ['', '1_', '2_', '3_']
labels = ['M, AoA', 'M+0.071, AoA+0.392', 'M+0.047, AoA-0.419', 'M-0.017, AoA +0.173']
colors = ['.5', '.7', '.9', '.25']
directory = '../../../data/noise/'
median = []
for i in range(len(designs)):
    df = pickle.load(open(directory + designs[i] + 'path_noise.p','rb'))
    df.reset_index(drop=True, inplace=True)
    sns.distplot(df['noise'], hist = False, kde = True,
                 kde_kws = {'linewidth': 2},
                 color = colors[i],
                 label = labels[i])
    median.append(np.median(df['noise']))
plt.xlabel('PLdB')
plt.ylabel('Density')

# Print medians
plt.figure()
x = [-0.017, 0, 0.047, 0.071]
y = [median[-1], median[0], median[2], median[1]]
plt.plot(x, y)
plt.scatter(x, y)
plt.xlabel('Change in Mach number')
plt.ylabel('PLdB')
# plt.plot([np.median(data['noise']),np.median(data['noise'])],[0, 0.2], color = '0.5', linewidth = 2)
# plt.plot([np.median(df['noise']),np.median(df['noise'])],[0, 0.2], color = 'k', linewidth = 2)
plt.show()