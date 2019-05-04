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
locations = ['72249']  # Corresponds to Fort Worth/Dallas

f = open('2018.p', 'rb')
data = pickle.load(f)
f.close()

plt.figure()
plt.plot(data['noise'])

average = []
minimum = []
maximum = []
std = []
for month in range(1, 13):
    noise_array = np.array(data['noise'])
    month_array = np.array(data['month'])
    average.append(np.average(noise_array[np.where(month_array == month)[0]]))
    minimum.append(noise_array[np.where(month_array == month)[0]].min)
    maximum.append(noise_array[np.where(month_array == month)[0]].max)
    std.append(np.std(noise_array[np.where(month_array == month)[0]]))

# altitudes = data['height']
# for i in range(len(data['height'])):
#     f = interp1d(data['height'], data['month'])

x = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
x_axis = range(len(x))

fig, ax = plt.subplots(1, 1)
ax.set_xticks(x_axis)  # set tick positions
# Labels are formated as integers:
ax.set_xticklabels(x)

ax.plot(x_axis, average)

ax.errorbar(x_axis, average, std, fmt='o', solid_capstyle='projecting')
# plt.fill_between(x, y3, y4, color='grey', alpha='0.5')
plt.show()
