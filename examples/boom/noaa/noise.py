import platform
import numpy as np

from weather.boom import boom_runner
from weather.scraper.noaa import process, output_for_sBoom


year = '2018'
month = '12'
day = '21'
hour = '12'
directory = '../../../matlab/'
filename = directory + year + month + day + '_' + hour + '.mat'

longitude = -100
latitude = 32
altitude = 50000

# Process weather data
data = process(filename)
sBoom_data, elevation = output_for_sBoom(data, longitude, latitude, altitude)

# Run sBoom
noise = boom_runner(sBoom_data, altitude, elevation)
print(noise)
