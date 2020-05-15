from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import pickle

directory = '../../../data/noise/'
f = open(directory + 'balloon/'+ 'radiosonde_database.p', 'rb')
data = pickle.load(f)
df = pd.DataFrame(data)
f.close()

print('Elevation vs PLdB', spearmanr(df['elevation'], df['noise']))
print('Average RH vs PLdB', spearmanr(df['average_rh'], df['noise']))
print('Maximum RH vs PLdB', spearmanr(df['max_rh'], df['noise']))

