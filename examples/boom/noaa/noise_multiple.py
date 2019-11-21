import platform
import numpy as np
import pickle

from weather.boom import boom_runner
from weather.scraper.noaa import process, output_for_sBoom


year = '2018'
month = '12'
hour = '12'
input_directory = '../../../matlab/'
output_directory = '../../../data/noise/'
for day in ['18', '19', '20', '21', '22']:
    filename = input_directory + year + month + day + '_' + hour + '.mat'

    alt_ft = 50000

    # Process weather data
    data = process(filename)
    data.noise = []
    for index in range(len(data.lonlat)):
        sBoom_data, altitude = output_for_sBoom(data, data.lonlat[index][0],
                                                data.lonlat[index][1], alt_ft)
        try:
            noise = boom_runner(sBoom_data, altitude)
        except:
            # Remove highest wind point in case of failure. Usually the reason
            sBoom_data[1] = sBoom_data[1][:-1]
            noise = boom_runner(sBoom_data, altitude)
        print(data.lonlat[index], noise)
        data.noise.append(noise)

    f = open(output_directory + year + month + day + '_' + hour + '_'
             + str(alt_ft) + ".p", "wb")
    pickle.dump(data, f)