import platform
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn, RegularGridInterpolator, interp1d

from weather.boom import boom_runner
from weather.scraper.noaa import process, output_for_sBoom


year = '2018'
month = '06'
day = '21'
hour = '00'
directory = '../../../matlab/'
filename = directory + year + month + day + '_' + hour + '.mat'

longitude = -107
latitude = 38
alt_ft = 50000

# Process weather data
data = process(filename)

# Setting up path
LAX = [33.9416, -118.4085]
JFK = [40.6413, -73.7781]
x = np.linspace(LAX[1], JFK[1], 30)
y = LAX[0] + (x-LAX[1])/(JFK[1]-LAX[1])*(JFK[0]-LAX[0])
path = np.array([y,x]).T

# Interpolate elevation
LON, LAT = np.meshgrid(data.lon, data.lat)
lonlat = np.array([LON.flatten(), LAT.flatten()]).T
lon = np.array(data.lon)
lat = np.array(data.lat)
elevation = np.flip(np.array(data.elevation),1)

f_elevation = RegularGridInterpolator((lat[::-1],lon), elevation)

# Preapre interpolation functions
humidity = np.flip(np.transpose(data.humidity, (1, 2, 0)), 1)
height = np.flip(np.transpose(data.height, (1, 2, 0)), 1)
temperature = np.flip(np.transpose(data.temperature, (1, 2, 0)), 1)
wind_x = np.flip(np.transpose(data.wind_x, (1, 2, 0)), 1)
wind_y = np.flip(np.transpose(data.wind_y, (1, 2, 0)), 1)

f_humidity = RegularGridInterpolator((lat[::-1],lon), humidity)
f_height = RegularGridInterpolator((lat[::-1],lon), height)
f_temperature = RegularGridInterpolator((lat[::-1],lon), temperature)
f_wind_x = RegularGridInterpolator((lat[::-1],lon), wind_x)
f_wind_y = RegularGridInterpolator((lat[::-1],lon), wind_y)

# Interpolating
path_elevation = f_elevation(path)
path_humidity = f_humidity(path)
path_height = f_height(path)
path_temperature = f_temperature(path)
path_wind_x = f_wind_x(path)
path_wind_y = f_wind_y(path)

# Contour plot
print(np.shape(path_humidity))
LON, HEIGHT = np.meshgrid(range(31), x)
plt.figure()
plt.contourf(HEIGHT, LON, path_humidity)
plt.xlabel('Longitude')
plt.ylabel('Height above ground')
clb = plt.colorbar()
clb.set_label('Relative Humidity')
plt.yticks([])
plt.show()

path_noise = []
for i in range(len(x)):
    # Consider elevation and round up (because of sboom input) for altitude 
    height_above_ground = np.around(path_height[i].tolist(), decimals=1)

    # Convert temperature from Kelvin to Farenheight
    temperature = (path_temperature[i] - 273.15) * 9/5. + 32
    weather = {}
    weather['wind'] = np.array([height_above_ground,
                                path_wind_x[i].tolist(),
                                path_wind_y[i].tolist()]).T

    weather['temperature'] = np.array([height_above_ground,
                                       temperature.tolist()]).T
    weather['humidity'] = np.array([height_above_ground,
                                    path_humidity[i].tolist()]).T

    for key in weather:
        weather[key] = weather[key].tolist()

    sBoom_data = [weather['temperature'], weather['wind'], weather['humidity']]
    altitude = alt_ft -path_elevation[i] 

    # Run sBoom
    noise = boom_runner(sBoom_data, altitude)
    print(noise)
    path_noise.append(noise)
plt.figure()
plt.plot(x, path_noise, 'r')
plt.ylabel('Perceived level in dB')
plt.xlabel('Longitude')
plt.show()