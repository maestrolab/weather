#!/usr/bin/env python
"""
convert dos linefeeds (crlf) to unix (lf)
usage: dos2unix.py 
"""
import pickle
import numpy, scipy.io

filename = '20180618_12_50000'
data = pickle.load(open(filename + '.p', 'rb'))
key_list = list(data.keys())
properties = ['temperature', 'pressure', 'humidity']
data_list = {}

# LatLon values
latlon_data = []
for key in key_list:
    latlon_data.append([float(x) for x in key.split(',')])
data_list['latlon'] = latlon_data

# Storing weather data

for property in properties:
    data_list[property] = []
for key in key_list:
    for property in properties:
        data_list[property].append([data[key][property]])

# Storing noise data
noise_list = []
for key in key_list:
    noise_list.append([data[key]['noise']])
data_list['noise'] = noise_list
scipy.io.savemat('./' + filename + '.mat', data_list)
