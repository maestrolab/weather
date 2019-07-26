from weather.scraper.balloon import balloon_scraper, process_data
from weather.filehandling import output_reader
from weather.boom import boom_runner
import numpy as np
import pickle

YEAR = '2018'
MONTH = '06'
DAY = '18'
HOUR = '00'
altitude = 50000
directory = './'

all_data = pickle.load(open('./72469_profiles.p','rb'))
log = open('log.txt', 'w')
noise = []
for i in range(2):
    temperature = all_data['temperature'][i]
    wind = all_data['wind'][i]
    humidity = all_data['humidity'][i]
    height_to_ground = all_data['height'][i]
    try:

        sBoom_data = [temperature, wind, humidity]

        noise = boom_runner(sBoom_data, height_to_ground,
                            nearfield_file='../../data/nearfield/25D_M16_RL5.p')

        all_data['noise'].append(noise)
    except(IndexError, ValueError) as e:
        print('Empty data')
        log.write('%i\n' % i)
all_data['noise'] = noise
log.close()
f = open('72469_noise.p', 'wb')
pickle.dump(all_data, f)
f.close()
