from weather.scraper.balloon import balloon_scraper, process_data
from weather.filehandling import output_reader
import numpy as np
import pickle

YEAR = '2018'
MONTH = '06'
DAY = '18'
HOUR = '00'
altitude = 50000
directory = './'
locations = ['71600', '71603', '71722', '71815', '72202', '72206', '72208',
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
n_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
# all_data = {'temperature': [], 'humidity': [], 'wind': [], 'month': [],
            # 'day': [], 'noise': [], 'height': []}
log = open('log.txt', 'w')

years = ['2000','2001','2002','2003','2004','2005','2006','2007','2008',
         '2009','2010','2011','2012','2013','2014','2015','2016','2017','2018']
years = [years[-2]]

path_name = './../humidity/autoencoder/balloon_data/2017/'

for location in locations:
    print(location)
    for year in years:
        YEAR = year
        print(YEAR)
        all_data = {'temperature': [], 'humidity': [], 'wind': [], 'month': [],
                    'day': [], 'noise': [], 'height': []}
        for month in range(1, len(n_days)+1):
            for day in range(1, n_days[month-1]+1):
                MONTH = '%02.f' % month
                DAY = '%02.f' % day
                try:
                    # filename = locations[0]+'.csv'

                    filename = path_name[:-5] + 'year_csv/%s_%s.csv' % (location, YEAR)
                    balloon_scraper(YEAR, MONTH, DAY, HOUR, directory, save=True,
                                    locations=[location], filename=filename)

                    data = output_reader(filename, header=['latitude', 'longitude',
                                                           'pressure', 'height',
                                                           'temperature', 'dew_point',
                                                           'humidity', 'mixr',
                                                           'wind_direction',
                                                           'wind_speed',
                                                           'THTA', 'THTE', 'THTV'],
                                         separator=',')
                    if max(np.array(data['height'])-data['height'][0]) > altitude * 0.3048:
                        sBoom_data, height_to_ground = process_data(data, altitude,
                                                                    directory='../data/weather/',
                                                                    outputs_of_interest=['temperature', 'height',
                                                                                         'humidity', 'wind_speed',
                                                                                         'wind_direction',
                                                                                         'latitude', 'longitude'],
                                                                    convert_celcius_to_fahrenheit=True)

                        [temperature, wind, humidity] = sBoom_data

                        print(month, day)
                        all_data['temperature'].append(temperature)
                        all_data['humidity'].append(humidity)
                        all_data['wind'].append(wind)
                        all_data['height'].append(data['height'])
                        all_data['month'].append(month)
                        all_data['day'].append(day)
                    else:
                        print('Not enough data')
                        log.write(YEAR + ', ' + MONTH + ', ' + DAY + '\n')
                except(IndexError, ValueError) as e:
                    print('Empty data')
                    log.write(YEAR + ', ' + MONTH + ', ' + DAY + '\n')

        year_name = '%s_%s' % (location, YEAR)
        f = open(path_name + year_name + '.p', 'wb')
        pickle.dump(all_data, f)
        f.close()

log.close()
# year_span = '_%i-%i' % (years[0], years[-1])
# f = open(locations[0] + year_span + '.p', 'wb')
# pickle.dump(all_data, f)
# f.close()
