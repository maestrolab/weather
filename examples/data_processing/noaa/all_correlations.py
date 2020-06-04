import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import pickle

directory = '../../../data/noise/' 
design = ''

pressures = np.array([100000, 97500, 95000, 92500, 90000, 85000, 80000, 75000, 70000,
                       65000, 60000, 55000, 50000, 45000, 40000, 35000, 30000, 25000,
                       20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 700, 500, 300,
                       200, 100])
altitude = 10.**5/2.5577*(1-(pressures/101325)**(1/5.2558)) / 0.3048
         
data = {'r_humidity':[], 'r_temperature':[], 'r_wind_x':[], 'r_wind_y':[], 'pressure':[], 'altitude':[]}
for index in range(9,22):
    print('index: ', index)
    filename = directory + design+ 'pressure_' +'%02d' % index + '.p'
    data['pressure'].append(pressures[index])
    data['altitude'].append(altitude[index])
    df = pickle.load(open(filename,'rb'))
    for key in ['temperature', 'humidity', 'wind_x', 'wind_y']:
        data['r_'+key].append(spearmanr(df[key], df['noise'])[0])
        
df = pd.DataFrame(data)
print(df)

plt.figure()

labels = ['Relative Humidity', 'Temperature (Negative correlation)', 'Wind (x-component)', 'Wind (y-component)']
keys = ['r_humidity', 'r_temperature', 'r_wind_x', 'r_wind_y']
for i in range(len(keys)):
    if i == 1:
        plt.plot(df['pressure'], -df[keys[i]], label=labels[i])
        plt.scatter(df['pressure'], -df[keys[i]])
    else:
        plt.plot(df['pressure'], df[keys[i]], label=labels[i])
        plt.scatter(df['pressure'], df[keys[i]])
plt.legend()
plt.ylim([0,1])
plt.xlim([min(df['pressure']), max(df['pressure'])])
plt.gca().invert_xaxis()
plt.ylabel('Spearman Rank Correlation Factor')
plt.xlabel('Atmospheric pressure (Pa)')
plt.show()
    
plt.figure()
# draw gridlines
labels = ['Relative Humidity', 'Temperature (Negative correlation)', 'Wind (x-component)', 'Wind (y-component)']
keys = ['r_humidity', 'r_temperature', 'r_wind_x', 'r_wind_y']
for i in range(len(keys)):
    if i==1:
        plt.plot(df['altitude'], -df[keys[i]], label=labels[i])
        plt.scatter(df['altitude'], -df[keys[i]])
    else:
        plt.plot(df['altitude'], df[keys[i]], label=labels[i])
        plt.scatter(df['altitude'], df[keys[i]])
plt.legend()
plt.ylim([0,1])
plt.xlim([min(df['altitude']), max(df['altitude'])])
plt.ylabel('Spearman Rank Correlation Factor')
plt.xlabel('Geopotential altitude (ft)')
# plt.xscale('log')
plt.show()