import matplotlib.pyplot as plt
from scipy import interpolate
import seaborn as sns
import numpy as np
import pickle
import pandas as pd
# from Lazzara
# lazzara = np.genfromtxt('../../data/msc/lazzara_pdf.csv', delimiter=',')
# d_lazzara = lazzara[lazzara[:,0].argsort()]

# x,y = d_lazzara.T
# plt.figure(figsize=(10,6))
# plt.plot(x,y, label='Lazzara et al.')

# From contour plots
day = '21'
year = '2018'
alt_ft = 50000
min_noise = 64
max_noise = 82
n_noise = 10
step = (max_noise - min_noise)/n_noise
bins = np.arange(min_noise, max_noise + step, step)
directory = '../../data/noise/'

d_contour = []
month_labels = ['June 2018', 'December 2018']
month_linestyle = ['--', '-.']
months = ['06', '12']
for i in range(2):
    month_label = month_labels[i]
    month = months[i]
    noise_day = []
    for hour in ['00','12']:
        # Get noise data
        filename = directory + year + month + day + '_' + hour + '_' + str(alt_ft) + ".p"
        data = pickle.load(open(filename, 'rb'))
        
        # Get rid of NaN
        array = np.ma.masked_invalid(np.array(data.noise).reshape(data.lon_grid.shape))

        #get only the valid values
        lon1 = data.lon_grid[~array.mask]
        lat1 = data.lat_grid[~array.mask]
        newarr = array[~array.mask]

        noise = interpolate.griddata((lon1, lat1), newarr.ravel(),
                                  (data.lon_grid, data.lat_grid),
                                     method='cubic')
        noise = noise.flatten()
        noise_day += list(noise)
    sns.distplot(noise_day, hist = False, kde = True,
                 kde_kws = {'linestyle': month_linestyle[i]},
                 color = 'k',
                 label = 'Forecast, North America, ' + month_label)
    print(month, len(noise[noise > 75 ])/len(noise))
    print('Forecast, North America, ' + month_label, len(noise_day))
# from radiosonde   

locations = ['72249', '72469']  # '72249' Corresponds to Fort Worth/Dallas 
labels = ['Radiosonde, Dallas, 2018', 'Radiosonde, Denver, 2018']
colors = ['b', 'r']
all_data = []
for i in range(2):
    
    f = open(directory + locations[i] + '_new.p', 'rb')
    data = pickle.load(f)
    f.close()
    all_data += list(data['noise'])
    sns.distplot(data['noise'], hist = False, kde = True,
                 label = labels[i], color=colors[i])
    print(labels[i], len(data['noise']))
# from radiosonde   
f = open(directory + 'balloon/'+ 'radiosonde_database.p', 'rb')
data = pickle.load(f)
f.close()
print('Radiosonde (2018) ', len(data['noise']))
sns.distplot(data['noise'], hist = False, kde = True,
             label = 'Radiosonde (2018)', color='0.5')
                 
# From path data 
df = pickle.load(open(directory + 'path_noise.p','rb'))
df.reset_index(drop=True, inplace=True)
print(78.7, len(df['noise'][df['noise'] > 78.7 ])/len(df['noise']))
print(75, len(df['noise'][df['noise'] > 75 ])/len(df['noise']))
sns.distplot(df['noise'], hist = False, kde = True,
             kde_kws = {'linewidth': 2},
             color = 'k',
             label = 'Forecast, Path, 2018 at UTC 12:00')

print('Forecast, Path, 2018 at UTC 12:00', len(df['noise'])) 
plt.xlabel('PLdB')
plt.ylabel('Density')

# Print medians
print('Radiosonde median: ', np.median(data['noise']))
print('Forecast median: ', np.median(df['noise']))
print('Radiosonde mean: ', np.mean(data['noise']))
print('Forecast mean: ', np.mean(df['noise']))
print('Radiosonde 99 CI: ', pd.DataFrame(data)['noise'].quantile(0.005), pd.DataFrame(data)['noise'].quantile(0.995))
print('Forecast 99 CI: ', df['noise'].quantile(0.005), df['noise'].quantile(0.995))

plt.plot([np.mean(data['noise']),np.median(data['noise'])],[0, 0.2], color = '0.5', linewidth = 2)
plt.plot([np.mean(df['noise']),np.median(df['noise'])],[0, 0.2], color = 'k', linewidth = 2)
plt.show()