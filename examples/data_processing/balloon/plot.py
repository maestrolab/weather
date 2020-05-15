import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
import pickle

from weather.scraper.balloon import balloon_scraper, process_data
from weather.filehandling import output_reader
from weather.boom import boom_runner


altitude = 50000
directory = '../../../data/noise/'
locations = ['72469']  # '72249' Corresponds to Fort Worth/Dallas '72469'

f = open(directory + locations[0] + '_new.p', 'rb')
data = pickle.load(f)
f.close()

# df = pd.DataFrame(data)
# count = 0
# for i in range(len(data['noise'])):
    # alt, property = np.array(data['humidity'][i]).T
    # print(max(alt))
    # if max(alt) < altitude * 0.3048:
        # count +=1
# print(count)
# BRAKE

df = pd.DataFrame(data)
q25 = df.groupby('month')['noise'].quantile(0.05)
q75 = df.groupby('month')['noise'].quantile(0.95)
q005 = df.groupby('month')['noise'].quantile(0.005)
q995 = df.groupby('month')['noise'].quantile(0.995)
average = []
minimum = []
maximum = []
median = []
std = []
data_per_month = []
for month in range(1, 13):
    noise_array = np.array(data['noise'])
    month_array = np.array(data['month'])
    average.append(np.average(noise_array[np.where(month_array == month)[0]]))
    median.append(np.median(noise_array[np.where(month_array == month)[0]]))
    minimum.append(noise_array[np.where(month_array == month)[0]].min())
    maximum.append(noise_array[np.where(month_array == month)[0]].max())
    std.append(np.std(noise_array[np.where(month_array == month)[0]]))
    data_per_month.append(noise_array[np.where(month_array == month)[0]])

q25_total = df['noise'].quantile(0.005)
q75_total = df['noise'].quantile(0.995)

altitudes = np.linspace(data['height'][0][0], altitude * 0.3048)
print('Above standard at Dallas: ', len(df['noise'][df['noise'] > 78.83 ])/len(df['noise']))
print('Above standard at sea-level: ', len(df['noise'][df['noise'] > 78.76])/len(df['noise']))
print(min(minimum), q25_total, np.average(data['noise']), np.median(data['noise']), q75_total, max(maximum), np.std(data['noise']))
properties_av = []
properties_std = []
properties_median = []
for altitude_i in altitudes:
    properties_at_altitude = []
    for i in range(len(data['noise'])):
        alt, property = np.array(data['humidity'][i]).T
        f = interp1d(alt, property)
        # print(i,alt)
        # print(altitude_i, min(data['height'][i]),max(data['height'][i]))
        properties_at_altitude.append(f(altitude_i))
    properties_av.append(np.average(properties_at_altitude))
    properties_median.append(np.median(properties_at_altitude))
    properties_std.append(np.std(properties_at_altitude))

plt.figure()
for i in range(len(np.array(data['noise']))):
    alt, property = np.array(data['humidity'][i]).T
    plt.plot(property, alt, 'k', alpha=0.05)
# plt.plot(properties_av, altitudes, 'r')
plt.plot(properties_av, altitudes, 'r')
# plt.plot(np.array(properties_av) + np.array(properties_std)/2., altitudes, '--r')
# plt.plot(np.array(properties_av) - np.array(properties_std)/2., altitudes, '--r')
plt.ylabel('Altitude (m)')
plt.xlabel('Relative humidity (%)')
plt.ylim(0, 35000)
plt.xlim(0, 100)
plt.show()

# Error plot
x = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
x_axis = range(0, len(x) * 2, 2)

fig, ax = plt.subplots(1, 1)
ax.set_xticks(x_axis)  # set tick positions
# Labels are formated as integers:
ax.set_xticklabels(x)

quantiles = np.squeeze(np.array([[median - q25.values , q75.values-median]]))

eb = ax.errorbar(x_axis, median, yerr=quantiles, marker='', color='k', capsize=5,
                 elinewidth=2,
                 markeredgewidth=2, ecolor='k',  ls='--')
plt.scatter(x_axis, median, c='k')
# eb[-1][0].set_linestyle('-- ')
# plt.fill_between(x, y3, y4, color='grey', alpha='0.5')
plt.ylim(68, 81)
print(maximum)
plt.plot(x_axis, q995, 'k', linestyle = '--', label = 'Maximum')
plt.plot(x_axis, q005, 'k', linestyle = '-.', label = 'Minimum')
plt.xlabel('Time in 2018')
plt.ylabel('Perceived level in dB (PLdB)')
plt.show()
