import platform
import numpy as np

from weather.boom import boom_runner
from weather.scraper.noaa import process, output_for_sBoom


year = '2018'
month = '06'
day = '18'
hour = '12'
directory = '../../../matlab/'
filename = directory + year + month + day + '_' + hour + '.mat'

longitude = -107
latitude = 38
alt_ft = 50000

# Process weather data
data = process(filename)

sBoom_data, altitude = output_for_sBoom(data, longitude, latitude, alt_ft)

# Run sBoom
noise = boom_runner(sBoom_data, altitude)
print(noise)
