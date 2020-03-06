import platform
import numpy as np
import pickle

from weather.boom import boom_runner
from weather.scraper.noaa import process, output_for_sBoom


year = '2018'
month = '06'
day = '21'
hour = '00'
input_directory = '../../../matlab/'
output_directory = '../../../data/noise/'
filename = input_directory + year + month + day + '_' + hour + '.mat'

altitude = 50000

# Process weather data
data = process(filename)
data.noise = []
for index in range(len(data.lonlat)):
    sBoom_data, elevation = output_for_sBoom(data, data.lonlat[index][0],
                                            data.lonlat[index][1], altitude)
    try:
        noise = boom_runner(sBoom_data, altitude, elevation)
    except:
        # Remove highest wind point in case of failure. Usually the reason
        sBoom_data[1] = sBoom_data[1][:-1]
        try:
            noise = boom_runner(sBoom_data, altitude, elevation)
        except(FileNotFoundError):
            noise = np.nan
    print(data.lonlat[index], noise)
    data.noise.append(noise)

f = open(output_directory + year + month + day + '_' + hour + '_'
         + str(altitude) + ".p", "wb")
pickle.dump(data, f)
