import platform
import numpy as np

from weather.boom import boom_runner
from weather.scraper.noaa import process, output_for_sBoom
from weather.scraper.geographic import elevation_function

year = '2018'
month = '12'
day = '06'
hour = '12'
directory = '../../../matlab/'
filename = directory + year + month + day + '_' + hour + '.mat'

longitude = -105
latitude = 40
altitude = 50000

# Process weather data
data = process(filename)

sBoom_data, elevation = output_for_sBoom(data, longitude, latitude, altitude)
elevation = elevation_function([latitude],[longitude])['elev_feet'][0]
print(elevation)
# Run sBoom
noise = boom_runner(sBoom_data, altitude, elevation)
print(noise)
