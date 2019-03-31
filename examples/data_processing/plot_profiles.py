import numpy as np
from weather.boom import process_data
import matplotlib.pyplot as plt
import pickle

day = '18'
month = '06'
year = '2018'
hour = '12'
lat = 38
lon = -107
alt_ft = 45000.

# Extracting data from database
alt_m = alt_ft * 0.3048
data, altitudes = process_data(day, month, year, hour, alt_m,
                               directory='../../data/weather/')
key = '%i, %i' % (lat, lon)
weather_data = data[key]

# Height to ground (HAG)
index = list(data.keys()).index(key)
height_to_ground = altitudes[index] / 0.3048

# Plotting
# Temperature
height, temperature = np.array(weather_data['temperature']).T
plt.figure(1)
plt.plot(temperature, height)
degree_sign = u'\N{DEGREE SIGN}'
plt.xlabel('Temperature (%sC)' % degree_sign, fontsize=12)
plt.ylabel('Height (m)', fontsize=12)
plt.title("Height vs Temperature at %.2s %sN, %.4s %sW" %
          (lat, degree_sign, lon, degree_sign), fontsize=16)

# Pressure
height, pressure = np.array(weather_data['pressure']).T
plt.figure(2)
plt.plot(pressure, height)
plt.xlabel('Pressure (hPa)', fontsize=12)
plt.ylabel('Height (m)', fontsize=12)
plt.title("Height vs Pressure at %.2s %sN, %.4s %sW" %
          (lat, degree_sign, lon, degree_sign), fontsize=16)

# Humidity
height, humidity = np.array(weather_data['humidity']).T
plt.figure(3)
plt.plot(humidity, height)
plt.xlabel('Relative Humidity (%)', fontsize=12)
plt.ylabel('Height (m)', fontsize=12)
plt.title("Height vs Relative Humidity at %.2s %sN, %.4s %sW" %
          (lat, degree_sign, lon, degree_sign), fontsize=16)

# Wind in the X-direction
height, wind_x = np.array(weather_data['wind_x']).T
plt.figure(4)
plt.plot(wind_x, height)
plt.xlabel('Wind Direction (%s)' % degree_sign, fontsize=12)
plt.ylabel('Height (m)', fontsize=12)
plt.title("Height vs Wind Direction at %.2s %sN, %.4s %sW" %
          (lat, degree_sign, lon, degree_sign), fontsize=16)

# Wind in the Y-direction
height, wind_y = np.array(weather_data['wind_y']).T
plt.figure(5)
plt.plot(wind_y, height)
plt.xlabel('Wind Speed (kn)', fontsize=12)
plt.ylabel('Height (m)', fontsize=12)
plt.title("Height vs Wind Speed at %.2s %sN, %.4s %sW" %
          (lat, degree_sign, lon, degree_sign), fontsize=16)

plt.show()
