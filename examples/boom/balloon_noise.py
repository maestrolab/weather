from weather.scraper.balloon import balloon_scraper, process_data
from weather.filehandling import output_reader
from weather.boom import boom_runner
import pickle

YEAR = '2018'
MONTH = '06'
DAY = '21'
HOUR = '00'
altitude = 50000
directory = './'
locations = ['72469']  # 72249 Corresponds to Fort Worth/Dallas
balloon_scraper(YEAR, MONTH, DAY, HOUR, directory, save=True,
                locations=locations)
data = output_reader('./WBData.csv', header=['latitude', 'longitude', 'pressure', 'height',
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
                                            convert_celcius_to_fahrenheit=False)

[temperature, wind, humidity] = sBoom_data

noise = boom_runner(sBoom_data, height_to_ground,
                    nearfield_file='../../data/nearfield/25D_M16_RL5.p')

print(noise)
