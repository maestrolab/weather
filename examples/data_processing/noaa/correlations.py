import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import pickle

directory = '../../../data/noise/' 
df = pickle.load(open(directory + 'path_noise.p','rb'))

df.sort_values(by=['elevation'], inplace=True)

PLdB_max = df.groupby('elevation')['noise'].max()
PLdB_min = df.groupby('elevation')['noise'].min()
PLdB_delta = PLdB_max - PLdB_min
PLdB_mean = df.groupby('elevation')['noise'].mean()
print('Elevation vs PLdB', spearmanr(df['elevation'], df['noise']))
print('Average RH vs PLdB', spearmanr(df['average_rh'], df['noise']))
print('Maximum RH vs PLdB', spearmanr(df['max_rh'], df['noise']))


x = df['elevation'].unique()
print('Elevation vs PLdB_max', spearmanr(x, PLdB_max))
print('Elevation vs PLdB_mean', spearmanr(x, PLdB_mean))
print('Elevation vs PLdB_min', spearmanr(x, PLdB_min))
print('Elevation vs PLdB_delta', spearmanr(x, PLdB_delta))

plt.figure()
plt.plot(x, PLdB_max, label='max')
plt.plot(x, PLdB_mean, label='mean')
plt.plot(x, PLdB_min, label='min')
plt.legend()
plt.show()