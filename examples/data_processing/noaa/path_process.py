import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import pandas as pd
import pickle

def get_profiles(df, index, altitude, filter = False):
    month = df['month'][index]
    day = df['day'][index]
    noise = df['noise'][index]
    data = pickle.load(open(directory + '/path_' + year + month + day + '_' + hour + '_' + str(alt_ft) + '.p', 'rb'))

    weather = {}
    index_data = data['noise'].index(noise)
    elevation = data['elevation'][index_data]* 0.3048 # convert to m
    
    h = data['height'][index_data]
    
    if min(h) > elevation:
        below_elevation_bool = len(h)*[False]
        below_elevation_bool[1] = True
        above_elevation_bool = len(h)*[True]
        above_elevation_bool[0] = False
        below_elevation = h[1]
        above_elevation = h[0]
    else:
        below_elevation_bool = h < elevation
        above_elevation_bool = h > elevation
        below_elevation = h[below_elevation_bool].max()
        above_elevation = h[above_elevation_bool].min()

    below_altitude_bool = h < altitude
    above_altitude_bool = h > altitude
    below_altitude = h[below_altitude_bool].max()
    above_altitude = h[above_altitude_bool].min()
    
    for key in data:
        if key != 'noise' and key != 'elevation':
            d = data[key][index_data]
            if not filter:
                weather[key] = d
            else:
                # first value
                if key == 'height':
                    weather[key] = [elevation]
                else:
                    below_key = d[below_elevation_bool][-1]
                    above_key = d[above_elevation_bool][0]
                    f = interpolate.interp1d([below_elevation, above_elevation], [below_key, above_key], fill_value="extrapolate")
                    weather[key] = list(f([elevation]))
                # In-between
                weather[key] += list(d[above_elevation_bool & below_altitude_bool])

                # Last value
                if key == 'height':
                    weather[key]+= [altitude]
                else:
                    below_key = d[below_altitude_bool][-1]
                    above_key = d[above_altitude_bool][0]
                    f = interpolate.interp1d([below_altitude, above_altitude], [below_key, above_key])
                    weather[key] += list(f([altitude]))
    return(month, day, weather, noise)


def plot_profiles(weather, label, linestyle='-', color = 'b'):
    height = 0.001*np.array(weather['height'])
    plt.subplot(2,2,1)
    plt.plot(weather['temperature'], height, color, label=label, linestyle=linestyle)
    plt.ylabel('Altitude (km)')
    plt.xlabel('Temperature (K)')
    
    plt.subplot(2,2,2)
    plt.plot(weather['humidity'], height, color, label=label, linestyle=linestyle)
    plt.ylabel('Altitude (km)')
    plt.xlabel('Humidity (%)')

    plt.subplot(2,2,3)
    plt.plot(weather['wind_x'], height, color, label=label, linestyle=linestyle)
    plt.ylabel('Altitude (km)')
    plt.xlabel('X-velocity (m/s)')
    
    plt.subplot(2,2,4)
    plt.plot(weather['wind_y'], height, color, label=label, linestyle=linestyle)
    plt.ylabel('Altitude (km)')
    plt.xlabel('Y-velocity (m/s)')
    plt.legend()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
    
df = pickle.load(open('extreme.p','rb'))
df.reset_index(drop=True, inplace=True)
directory = '../../../data/noise/'
hour = '12'
year = '2018'
alt_ft = 50000
alt_m = alt_ft*0.3048 # convert to m

# Select Maximum
plt.figure(figsize=(8, 8))

locations = [1675.938210097911, 2711.8195861504664]
for location_i in locations:
    location = find_nearest(df['location'], location_i)
    print(location_i, location)
    minimum = df[df['location'] == location]['noise'].min()
    q25_total = df[df['location'] == location]['noise'].quantile(0.005)
    average = np.mean(df[df['location'] == location]['noise'])
    median = np.median(df[df['location'] == location]['noise'])
    q75_total = df[df['location'] == location]['noise'].quantile(0.995)
    maximum = df[df['location'] == location]['noise'].max()
    print(minimum, q25_total, average, median, q75_total, maximum)

max_i = df['noise'].idxmax()
month, day, weather, noise = get_profiles(df, max_i, alt_m)
print('Maximum', noise)
plot_profiles(weather, label = '%.2f (complete)' % noise, linestyle= '--', color='b')
month, day, weather, noise = get_profiles(df, max_i, alt_m, filter=True)
plot_profiles(weather, label = '%.2f (used)' % noise, color='b')
# Select Minimum

min_i = df['noise'].idxmin()
month, day, weather, noise = get_profiles(df, min_i, alt_m)
print('Minimum', noise)
plot_profiles(weather, label = '%.2f (complete)' % noise, linestyle= '--', color='r')
month, day, weather, noise = get_profiles(df, min_i, alt_m, filter = True)
plot_profiles(weather, label = '%.2f (used)' % noise, color='r')
plt.show()

