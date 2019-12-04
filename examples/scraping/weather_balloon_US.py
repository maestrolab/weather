from weather.scraper.balloon import balloon_scraper, process_data
from weather.filehandling import output_reader
from weather.boom import boom_runner
import numpy as np
import pickle

YEARS = ['2017', '2018']
MONTH = '06'
DAY = '18'
HOUR = '12'
n_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
altitude = 50000
US_locations = ['71600', '71603', '71722', '71815', '72202', '72206', '72208',
                '72210', '72214', '72215', '72230', '72233', '72235', '72240',
                '72248', '72249', '72251', '72261', '72265', '72274', '72293',
                '72305', '72317', '72318', '72327', '72340', '72357', '72363',
                '72364', '72365', '72376', '72388', '72393', '72402', '72403',
                '72426', '72440', '72451', '72456', '72469', '72476', '72489',
                '72493', '72501', '72518', '72520', '72528', '72558', '72562',
                '72572', '72582', '72597', '72632', '72634', '72645', '72649',
                '72659', '72662', '72672', '72681', '72694', '72712', '72747',
                '72764', '72768', '72776', '72786', '72797', '74389', '74455',
                '74494', '74560', '74646', '74794', '78016', '78073']

log = open('data_missing_log.txt', 'w')

# This example saves data to be used as training/testing data for machine
#   learning models
directory = '../../data/atmosphere_models/test_data/individual_locations/'

for location in US_locations:
    print(location)
    for YEAR in YEARS:
        print(YEAR)
        all_data = {'temperature': [], 'humidity': [], 'wind': [], 'month': [],
                    'day': [], 'height': []}
        filename = '%s_%s' % (location, YEAR)

        for month in range(1, len(n_days)+1):
            for day in range(1, n_days[month-1]+1):
                MONTH = '%02.f' % month
                DAY = '%02.f' % day
                try:
                    balloon_scraper(YEAR, MONTH, DAY, HOUR, directory, save=True,
                                    locations=[location], filename=filename+'.csv')

                    print(year, month, day)
                    all_data['temperature'].append(temperature)
                    all_data['humidity'].append(humidity)
                    all_data['wind'].append(wind)
                    all_data['height'].append(data['height'])
                    all_data['year'].append(year)
                    all_data['month'].append(month)
                    all_data['day'].append(day)
                except:
                    print('Not enough data')
                    log.write(YEAR + ', ' + MONTH + ', ' + DAY + '\n')

        f = open(directory + filename + '.p', 'wb')
        pickle.dump(all_data, f)
        f.close()

log.close()
