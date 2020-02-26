#!/usr/bin/env python
"""
convert dos linefeeds (crlf) to unix (lf)
usage: dos2unix.py 
"""
import pickle
import numpy as np
import scipy.io

filename = '20180621_12_50000'
data = pickle.load(open(filename + '.p', 'rb'))
n = 46*92
key_list = range(n)
properties = ['temperature', 'pressure', 'humidity']
data_list = {}

# LatLon values
data_list['latlon'] = data.lonlat[:,::-1]
height = data.height.reshape([31, n]).T
temperature = data.temperature.reshape([31, n]).T
humidity = data.humidity.reshape([31, n]).T
# pressure = data.pressure.reshape([31, n])
wind_x = data.wind_x.reshape([31, n]).T
wind_y = data.wind_y.reshape([31, n]).T

# Storing weather data
data_list['temperature'] = temperature
data_list['humidity'] = humidity
# data_list['pressure'] = np.array([height, pressure]).T
data_list['wind_x'] = wind_x
data_list['wind_y'] = wind_y
data_list['height'] = height
data_list['noise'] = data.noise
data_list['elevation'] = np.array(data.elevation).flatten().T

# Storing noise data
scipy.io.savemat('./' + filename + '.mat', data_list)
