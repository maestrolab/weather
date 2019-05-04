from weather.scraper.balloon import balloon_scraper, process_data
from weather.filehandling import output_reader
from weather.boom import boom_runner
import pickle

YEAR = '2018'
MONTH = '06'
DAY = '18'
HOUR = '00'
altitude = 50000
directory = './'
locations = ['72249']  # Corresponds to Fort Worth/Dallas
n_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
all_data = {'temperature': [], 'humidity': [], 'wind': [], 'month': [],
            'day': [], 'noise': []}
log = open('log.txt', 'w')
for month in range(1, len(n_days)+1):
    for day in range(1, n_days[month-1]+1):
        MONTH = '%02.f' % month
        DAY = '%02.f' % day
        try:
            balloon_scraper(YEAR, MONTH, DAY, HOUR, directory, save=True,
                            locations=locations)

            data = output_reader('./WBData.csv', header=['latitude', 'longitude',
                                                         'pressure', 'height',
                                                         'temperature', 'dew_point',
                                                         'humidity', 'mixr',
                                                         'wind_direction',
                                                         'wind_speed',
                                                         'THTA', 'THTE', 'THTV'],
                                 separator=',')

            sBoom_data, height_to_ground = process_data(data, altitude,
                                                        directory='../data/weather/',
                                                        outputs_of_interest=['temperature', 'height',
                                                                             'humidity', 'wind_speed',
                                                                             'wind_direction',
                                                                             'latitude', 'longitude'],
                                                        convert_celcius_to_fahrenheit=True)

            [temperature, wind, humidity] = sBoom_data

            noise = boom_runner(sBoom_data, height_to_ground,
                                nearfield_file='../../data/nearfield/25D_M16_RL5.p')

            print(month, day, noise)
            all_data['temperature'].append(temperature)
            all_data['humidity'].append(humidity)
            all_data['wind'].append(wind)
            all_data['month'].append(month)
            all_data['noise'].append(noise)
            all_data['day'].append(day)
        except(IndexError):
            print('Empty data')
            log.write(YEAR + ', ' + MONTH + ', ' + DAY + '\n')

log.close()
f = open('2018.p', 'wb')
pickle.dump(all_data, f)
f.close()
