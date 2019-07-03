'''Combines twister data into a single pickle file to be used with the
ClassifiedProfile class.'''

import pickle
from weather.scraper.twister import process_data

from misc_humidity import package_data, combine_profiles

day = '18'
month = '06'
year = '2018'
hour = '12_'
alt_ft = 45000.
alt = alt_ft * 0.3048

data, altitudes = process_data(day, month, year, hour, alt,
                               directory='./../../../../data/weather/twister/',
                               convert_celcius_to_fahrenheit=True)

path = './'
combine_profiles(data, day, month, year, hour, alt, profile_type='temperature',
                 path = path)
