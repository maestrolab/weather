from weather.scraper.balloon import balloon_scraper, process_data
from weather.filehandling import output_reader
from weather.scraper.geographic import elevation_function
from weather.boom import boom_runner_eq
import numpy as np
import pickle

YEAR = '2018'
HOUR = '12'
altitude = 50000
directory = './'

city = 'Denver'
if city == 'Denver':
    locations = ['72469']
    # city_elevation = elevation_function([39.7392], [-104.9903])['elev_feet']
elif city == 'Fort Worth':
    locations = ['72249']
    # city_elevation = elevation_function([32.7555], [-97.3308])['elev_feet']
 
locations = ['72469']  # 72249 Corresponds to Fort Worth/Dallas, 72469 is Denver
n_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
all_data = {'temperature': [], 'humidity': [], 'wind': [], 'month': [],
            'day': [], 'noise': [], 'height': [], 'elevation':[]}
log = open('log.txt', 'w')


for month in range(1, 3):
    for day in range(1, 32):
        MONTH = '%02.f' % month
        DAY = '%02.f' % day
        try:
            filename = locations[0]+'.csv'
            balloon_scraper(YEAR, MONTH, DAY, HOUR, directory, save=True,
                            locations=locations, filename=filename)

            data = output_reader(filename, header=['latitude', 'longitude',
                                                   'pressure', 'height',
                                                   'temperature', 'dew_point',
                                                   'humidity', 'mixr',
                                                   'wind_direction',
                                                   'wind_speed',
                                                   'THTA', 'THTE', 'THTV'],
                                 separator=',')
            if max(np.array(data['height'])-data['height'][0]) > altitude * 0.3048:
                sBoom_data, elevation = process_data(data, altitude,
                                                            directory='../data/weather/',
                                                            outputs_of_interest=['temperature', 'height',
                                                                                 'humidity', 'wind_speed',
                                                                                 'wind_direction',
                                                                                 'latitude', 'longitude'],
                                                            convert_celcius_to_fahrenheit=True)
                
                [temperature, wind, humidity] = sBoom_data

                noise = boom_runner_eq(sBoom_data, altitude, elevation)

                print(month, day, noise)
                all_data['temperature'].append(temperature)
                all_data['humidity'].append(humidity)
                all_data['wind'].append(wind)
                all_data['height'].append(data['height'])
                all_data['month'].append(month)
                all_data['noise'].append(noise)
                all_data['day'].append(day)
                all_data['elevation'].append(elevation)
            else:
                print('Not enough data')
                log.write(YEAR + ', ' + MONTH + ', ' + DAY + '\n')
        except(IndexError, ValueError, FileNotFoundError) as e:
            print('Empty data')
            log.write(YEAR + ', ' + MONTH + ', ' + DAY + '\n')

log.close()
f = open(locations[0] + '.p', 'wb')
pickle.dump(all_data, f)
f.close()
