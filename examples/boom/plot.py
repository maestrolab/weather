import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import pickle

from weather.scraper.balloon import balloon_scraper, process_data
from weather.filehandling import output_reader
from weather.boom import boom_runner


YEAR = '2018'
MONTH = '06'
DAY = '18'
HOUR = '00'
altitude = 50000
directory = './'
locations = ['72469']  # Corresponds to Fort Worth/Dallas

f = open(locations[0] + '.p', 'rb')
data = pickle.load(f)
f.close()

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
    minimum.append(noise_array[np.where(month_array == month)[0]].min)
    maximum.append(noise_array[np.where(month_array == month)[0]].max)
    std.append(np.std(noise_array[np.where(month_array == month)[0]]))
    data_per_month.append(noise_array[np.where(month_array == month)[0]])

altitudes = np.linspace(0, altitude * 0.3048 - data['height'][0][0])
properties_av = []
properties_std = []
properties_median = []
for altitude in altitudes:
    properties_at_altitude = []
    for i in range(len(data['noise'])):
        alt, property = np.array(data['humidity'][i]).T
        f = interp1d(alt, property)
        properties_at_altitude.append(f(altitude))
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


eb = ax.errorbar(x_axis, average, std, marker='', color='k', capsize=5,
                 elinewidth=2,
                 markeredgewidth=2, ecolor='k',  ls='--')
plt.scatter(x_axis, average, c='k')
# eb[-1][0].set_linestyle('-- ')
# plt.fill_between(x, y3, y4, color='grey', alpha='0.5')
plt.ylim(min(data['noise']), max(data['noise']))
plt.xlabel('Time in 2018')
plt.ylabel('Perceived level in dB (PLdB)')
plt.show()
