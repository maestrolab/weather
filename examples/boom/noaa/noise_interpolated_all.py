import platform
import pickle
import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
from scipy.interpolate import interpn, RegularGridInterpolator, interp1d

from weather.scraper.geographic import haversine, elevation_function
from weather.boom import boom_runner_eq
from weather.scraper.noaa import process, output_for_sBoom


directory = '../../../matlab/'
output_directory = '../../../data/noise/'
eq_area_file = 'Mach1.671_Alpha0.392_HL5.dat'

# Includes College Station
# lat_cities = [47.6062, 43.6150, 39.7392, 30.6280, 25.7617, ]
# lon_cities = [-122.3321, -116.2023, -104.9903, -96.3344, -80.1918]
# Included Fort Worth/Dallas
lat_cities = [47.6062, 43.6150, 39.7392, 32.7555, 25.7617, ]
lon_cities = [-122.3321, -116.2023, -104.9903, -97.3308, -80.1918]

lat_all = []
lon_all = []
distance_all = [0]
distance_cities = [0]
for i in range(len(lat_cities)-1):
    j = i+1
    lon_path = np.linspace(lon_cities[i], lon_cities[j], 25)
    lat_path = lat_cities[i] + (lon_path-lon_cities[i])/(lon_cities[j]-lon_cities[i])*(lat_cities[j]-lat_cities[i])
    lon_all += list(lon_path)
    lat_all += list(lat_path)

for i in range(len(lat_all)-1):
    j = i+1
    distance_all.append(distance_all[-1] + haversine(lon_all[i], lat_all[i],
                                                     lon_all[j], lat_all[j]))
for i in range(len(lat_cities)-1):
    j = i+1
    distance_cities.append(distance_cities[-1] + haversine(lon_cities[i], lat_cities[i],
                                                     lon_cities[j], lat_cities[j]))

# Setting up path
path = np.array([lat_all,lon_all]).T
path_elevation = elevation_function(lat_all, lon_all)['elev_feet']
   
for month in range(1,13):
    month = '%02i' % month
    for day in range(1,31):
        try:
            day = '%02i' % day
            print(month, day)
            year = '2018'
            hour = '12'
            filename = directory + year + month + day + '_' + hour + '.mat'

            alt_ft = 50000

            # Process weather data
            data = process(filename)

            # Interpolate elevation
            LON, LAT = np.meshgrid(data.lon, data.lat)
            lonlat = np.array([LON.flatten(), LAT.flatten()]).T
            lon = np.array(data.lon)
            lat = np.array(data.lat)

            # Prepare interpolation functions
            humidity = np.flip(np.transpose(data.humidity, (1, 2, 0)), 0)
            height = np.flip(np.transpose(data.height, (1, 2, 0)), 0)
            temperature = np.flip(np.transpose(data.temperature, (1, 2, 0)), 0)
            wind_x = np.flip(np.transpose(data.wind_x, (1, 2, 0)), 0)
            wind_y = np.flip(np.transpose(data.wind_y, (1, 2, 0)), 0)

            f_humidity = RegularGridInterpolator((lat[::-1],lon), humidity)
            f_height = RegularGridInterpolator((lat[::-1],lon), height)
            f_temperature = RegularGridInterpolator((lat[::-1],lon), temperature)
            f_wind_x = RegularGridInterpolator((lat[::-1],lon), wind_x)
            f_wind_y = RegularGridInterpolator((lat[::-1],lon), wind_y)

            # Interpolating
            
            path_humidity = f_humidity(path)
            path_height = f_height(path)
            path_temperature = f_temperature(path)
            path_wind_x = f_wind_x(path)
            path_wind_y = f_wind_y(path)


            path_noise = []
            for i in range(len(lon_all)):
                # Consider elevation and round up (because of sboom input) for altitude
                elevation = path_elevation[i]
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
                altitude = alt_ft

                # Run sBoom
                try:
                    noise = boom_runner_eq(sBoom_data, altitude, elevation, nearfield_file=eq_area_file)
                except:
                    # Remove highest wind point in case of failure. Usually the reason
                    sBoom_data[1] = sBoom_data[1][:-1]
                    try:
                        noise = boom_runner_eq(sBoom_data, altitude, elevation, nearfield_file=eq_area_file)
                    except(FileNotFoundError):
                        noise = np.nan
                print(noise)
                path_noise.append(noise)
            output = {'noise': path_noise,
                      'humidity': path_humidity,
                      'wind_x': path_wind_x,
                      'wind_y': path_wind_y,
                      'temperature': path_temperature,
                      'elevation': path_elevation,
                      'height': path_height}
            f = open(output_directory + 'path_' + year + month + day + '_' + hour + '_'
                     + str(alt_ft) + ".p", "wb")
            pickle.dump(output, f)
            f.close()
        except(FileNotFoundError):
            pass
