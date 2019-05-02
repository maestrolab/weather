from weather.scraper.balloon import balloon_scraper
from weather.scraper.twister import process_data
import pickle

YEAR = '2018'
MONTH = '06'
DAY = '18'
HOUR = '00'
altitude = 50000
directory = '../../data/weather/balloon/'
locations = ['72249']  # Corresponds to Fort Worth/Dallas
data = balloon_scraper(YEAR, MONTH, DAY, HOUR, directory, save=True,
                       locations=locations)
print(data)
data, altitudes = process_data(DAY, MONTH, YEAR, HOUR, altitude,
                               directory='../data/weather/',
                               outputs_of_interest=['temperature', 'height',
                                                    'humidity', 'wind_speed',
                                                    'wind_direction', 'pressure',
                                                    'latitude', 'longitude'],
                               convert_celcius_to_fahrenheit=False,
                               data=data)

# all_data = {'latitude': [], 'longitude': [], 'pressure': [], 'height': [],
#             'temperature': [], 'humidity': [], 'wind_direction': [],
#             'wind_speed':
