import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import pickle

from weather.scraper.balloon import balloon_scraper, process_data
from weather.filehandling import output_reader
from weather.boom import boom_runner


def process_noise(data):
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
    return average, median, minimum, maximum, std, data_per_month


def process_properties(data, property, altitude):
    altitudes = np.linspace(0, altitude * 0.3048 - data['height'][0][0])
    properties_av = []
    properties_std = []
    properties_median = []
    for altitude in altitudes:
        properties_at_altitude = []
        for i in range(len(data['noise'])):
            alt, property = np.array(data[property][i]).T
            print(altitude, min(alt), max(alt))
            f = interp1d(alt, property)
            properties_at_altitude.append(f(altitude))
        properties_av.append(np.average(properties_at_altitude))
        properties_median.append(np.median(properties_at_altitude))
        properties_std.append(np.std(properties_at_altitude))
    return properties_av, properties_median, properties_std


def setBoxColors(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


YEAR = '2018'
MONTH = '06'
DAY = '18'
HOUR = '00'
altitude = 50000
directory = './'
locations = ['72249', '72469']  # Corresponds to Fort Worth/Dallas

x = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
x_axis = range(0, len(x) * 2, 2)
fig1, ax1 = plt.subplots(1, 1)
ax1.set_xticks(x_axis)  # set tick positions
# Labels are formated as integers:
ax1.set_xticklabels(x)
plt.xlabel('Time in 2018')
plt.ylabel('Perceived level in dB (PLdB)')

fig2, ax2 = plt.subplots(1, 1)
plt.xlabel('Time in 2018')
plt.ylabel('Perceived level in dB (PLdB)')


colors = ['b', 'r']
offset = [-0.4, 0.]
for i in range(len(locations)):
    location = locations[i]
    color = colors[i]
    f = open(location + '.p', 'rb')
    data = pickle.load(f)
    f.close()

    [average, median, minimum, maximum, std, data_per_month] = process_noise(data)
# Error plot

    ax1.errorbar(x_axis, average, std, marker='', color=color, capsize=5,  # , fmt='none'
                 elinewidth=2,
                 markeredgewidth=2, ecolor=color,  ls='--')
    ax1.scatter(x_axis, median, c=color)

    bp = ax2.boxplot(data_per_month, positions=np.array(
        range(len(data_per_month)))*1.0+offset[i], showfliers=False)
    setBoxColors(bp, color)

plt.xlim(-1, len(x))
plt.show()
