import weather

import requests
from bs4 import BeautifulSoup
import datetime
import pickle
import copy
import math
import csv
import numpy as np
from weather import convert_to_fahrenheit, windToXY, makeFloats


def balloon_scraper(YEAR, MONTH, DAY, HOUR, directory='./', save=False,
                    locations=None, filename='database.csv'):

    FROM = DAY + HOUR
    TO = DAY + HOUR

    # LIST OF LOCATIONS ACROSS NA
    if locations is None:
        locations = ['03953', '04220', '04270', '04360', '08508', '70133',
                     '70200', '70219', '70231', '70261', '70273', '70316',
                     '70326', '70350', '70398', '70414', '71043', '71081',
                     '71109', '71119', '71126', '71600', '71603', '71722',
                     '71811', '71815', '71816', '71823', '71836', '71845',
                     '71867', '71906', '71907', '71908', '71909', '71913',
                     '71924', '71925', '71926', '71934', '71945', '71957',
                     '71964', '72201', '72202', '72206', '72208', '72210',
                     '72214', '72215', '72230', '72233', '72235', '72240',
                     '72248', '72249', '72251', '72261', '72265', '72274',
                     '72293', '72305', '72317', '72318', '72327', '72340',
                     '72357', '72363', '72364', '72365', '72376', '72388',
                     '72393', '72402', '72403', '72426', '72440', '72451',
                     '72456', '72469', '72476', '72489', '72493', '72501',
                     '72518', '72520', '72528', '72558', '72562', '72572',
                     '72582', '72597', '72632', '72634', '72645', '72649',
                     '72659', '72662', '72672', '72681', '72694', '72712',
                     '72747', '72764', '72768', '72776', '72786', '72797',
                     '74005', '74389', '74455', '74494', '74560', '74646',
                     '74794', '76256', '76394', '76458', '76526', '76595',
                     '76612', '76644', '76654', '76679', '76805', '78016',
                     '78073', '78384', '78397', '78486', '78526', '78583',
                     '78807', '78897', '78954', '78970', '91285', '80222',
                     '82022', '91165', '91285']

    f = open(directory+filename, 'w')
    counter_x = 0
    counter_filter = 0

    all_data = {'latitude': [], 'longitude': [], 'pressure': [], 'height': [],
                'temperature': [], 'humidity': [], 'wind_direction': [],
                'wind_speed': []}
    for location in locations:
        valid_data = True
        counter_x += 1
        page = 'http://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST&YEAR=' + \
            YEAR + '&MONTH=' + MONTH + '&FROM=' + FROM + '&TO=' + TO + \
            '&STNM=' + location

        # print(counter_x)
        page = requests.get(page)
        page.content
        soup = BeautifulSoup(page.content, 'html.parser')
        soup = soup.text.split('\n')

        # For loop will find two lines that start with '-' and start processing from there
        counter = 0

        if len(soup) > 100:
            while counter != 2:
                # print(len(soup))
                if len(soup[0]) != 0:
                    if soup[0][0] == '-':
                        counter += 1
                soup.pop(0)
            new_soup = []

            # For loop will add new lines until finds a string starting with 'S'
            i = 0
            while soup[i][0] != 'S':
                new_soup.append(soup[i])
                i += 1
            after_soup = soup[i:]
            soup = new_soup

            # Replace all multiple spacing for a single one
            for i in range(len(soup)):
                while '  ' in soup[i]:
                    soup[i] = soup[i].replace('  ', ' ')
                    if soup[i][0] == ' ':
                        soup[i] = soup[i][1:]

            # Search for latitude and longitude
            for i in range(len(after_soup)):
                try:
                    if 'Station latitude:' in after_soup[i]:
                        temp = after_soup[i].replace('Station latitude:', '')
                        temp = temp.replace('\n', '')
                        latitude = float(temp)
                    if 'Station longitude:' in after_soup[i]:
                        temp = after_soup[i].replace('Station longitude:', '')
                        temp = temp.replace('\n', '')
                        longitude = float(temp)
                except:
                    valid_data = False
            # Replace all spaces for commas
            if valid_data:
                for i in range(len(soup)):
                    while ' ' in soup[i]:
                        soup[i] = soup[i].replace(' ', ',')
                        if soup[i][0] == ',':
                            soup[i] = soup[i][1:]
                    if soup[i].count(',') == 10:
                        # counter_filter += 1
                        # if counter_filter == 8:
                        #     counter_filter = 0
                        # if counter_filter == 0:
                        f.write(str(latitude) + ',' + str(longitude) +
                                ',' + soup[i]+'\n')

    f.close()

    return all_data


def process_data(all_data, altitude,
                 directory='../data/weather/',
                 outputs_of_interest=['temperature', 'height',
                                      'humidity', 'wind_speed',
                                      'wind_direction', 'pressure'],
                 convert_celcius_to_fahrenheit=True):
    ''' process_data makes a dictionary output that contains the lists
    specified by the strings given in outputs_of_interest
    '''
    # Reading data for selected properties
    if outputs_of_interest == 'all':
        output = all_data
    else:
        output = {}

        for key in outputs_of_interest:
            output[key] = copy.deepcopy(all_data[key])

    # Make everything floats
    for key in outputs_of_interest:
        output[key] = makeFloats(output[key])

    # Convert temperature to fahrenheit
    if convert_celcius_to_fahrenheit:
        output['temperature'] = convert_to_fahrenheit(output['temperature'])

    # Convert wind data
    wind_x, wind_y = windToXY(output['wind_speed'],
                              output['wind_direction'])
    output['wind_x'] = wind_x
    output['wind_y'] = wind_y
    ground_altitude = output['height'][0]
    output['height'] = [x - ground_altitude for x in output['height']]
    output.pop('wind_speed', None)
    output.pop('wind_direction', None)

    # # Prepare for sBOOM
    # data = {}
    data = {}

    data['temperature'] = np.array([output['height'],
                                    output['temperature']]).T
    data['humidity'] = np.array([output['height'],
                                 output['humidity']]).T
    data['wind'] = np.array([output['height'],
                             output['wind_x'],
                             output['wind_y']]).T
    for key in data.keys():
        data[key] = np.unique(data[key], axis=0).tolist()
    #     lat = output['latitude']
    #     lon = output['longitude']
    #     height = output['height']
    #     if key not in ['latitude', 'longitude', 'height']:
    #         data, ground_altitudes = output_for_sBoom(output[key],
    #                                                   key, altitude, lat,
    #                                                   lon, height, data)
    return [data['temperature'], data['wind'], data['humidity']], altitude - ground_altitude / 0.3048


def output_for_sBoom(li, keyName, ALT, lat, lon, height, data):
    '''output_for_sBoom takes a weather variable list, list keyName, and
    a max altitude (ALT) as user defined inputs. It also requires the
    existance of a dictionary data, and the lat, lon, and height lists
    from the openPickle function. Using these, it makes a dictionary
    with first key being a lat,lon point and second key being the
    name of the weather variable.
    '''
    temp_height = []
    temp_li = []
    temp_combo_li = []
    d = copy.deepcopy(data)
    k = 0
    i = 0
    ground_level = 0
    ground_altitudes = []
    # ground_level = 0
    # ground_altitudes = []
    while i < len(lat):
        if i > 0:
            # appending to mini-list
            if lat[i] == 0:
                temp_height.append(height[i] - ground_level)
                temp_li.append(li[i])
                k += 1
                i += 1
            else:
                # combining height and weather mini lists for storage
                temp_combo_li = combineLatLon(temp_height, temp_li)

                # making sure first two heights aren't the same
                if temp_combo_li[0][0] == temp_combo_li[1][0]:
                    temp_combo_li.pop(0)

                # TEMPORARY TEST-forcing list to be a certain length
                # while len(temp_combo_li) > 20:
                    # temp_combo_li.pop()

                # getting to next latlon value in big list if not already there
                # while lat[i] == 0:
                    # i += 1
                    # k += 1

                # key is location of previous latlon in big list
                key = '%i, %i' % (lat[i-k], lon[i-k])

                # appending mini-list to dictionary at latlon key
                if d:
                    data[key][keyName] = temp_combo_li
                else:
                    data[key] = {keyName: temp_combo_li}

                # clearing mini-list and restarting
                temp_height = []
                temp_li = []
                temp_combo_li = []
                k = 0
                temp_height.append(height[i])
                ground_level = temp_height[0]
                ground_altitudes.append(ALT - ground_level)
                temp_height[0] = temp_height[0] - ground_level
                temp_li.append(li[i])
                k += 1
                i += 1
        # getting first element in big list
        else:
            temp_height.append(height[i])
            ground_level = temp_height[0]

            ground_altitudes = [ALT - ground_level]
            temp_height[0] = temp_height[0] - ground_level
            temp_li.append(li[i])
            k += 1
            i += 1

    # getting data from final mini-list
    temp_combo_li = combineLatLon(temp_height, temp_li)
    # making sure first two heights aren't the same
    if temp_combo_li[0][0] == temp_combo_li[1][0]:
        temp_combo_li.pop(0)

    # while len(temp_combo_li) > 20:
        # temp_combo_li.pop()

    # dictionary key
    key = '%i, %i' % (lat[i-k], lon[i-k])

    # making dictionary
    if d:
        data[key][keyName] = temp_combo_li
    else:
        data[key] = {keyName: temp_combo_li}

    return data, ground_altitudes


def combineLatLon(lat, lon):
    '''combineLatLon takes a list of latitudes and a list of longitudes
    that are the same length and combines them into a double list.
    '''
    w_latlon = []
    for i in range(len(lat)):
        w_latlon.append([lat[i], lon[i]])

    return w_latlon


def process_database(filename, variable_name='noise', transformation=None):
    ''' Variable names = 'noise', 'temperature', 'wind_x', 'wind_y', 'pressure',
        '''
    noise_data = pickle.load(open(filename + '.p', 'rb'))

    lat = []
    lon = []
    z = []
    latlon = copy.deepcopy(list(noise_data.keys()))

    for i in range(len(latlon)):
        latlon_temp = [int(s) for s in latlon[i].split(',')]
        lat.append(latlon_temp[0])
        lon.append(latlon_temp[1])
        z.append(noise_data[latlon[i]][variable_name])

    # Make lists into arrays to graph
    lon = np.array(lon)
    lat = np.array(lat)
    z = np.array(z)
    if transformation is not None:
        z = transformation(z)
    return np.vstack([lon, lat, z]).T


def appendToDictionary(latitude, longitude, all_data, soup):
    ''' appendToDictionary appends the data scraped from twisterdata.com
    to a dictionary for later use in this repository.
    '''
    all_data['latitude'].append(latitude)
    all_data['longitude'].append(longitude)

    prevLength = len(all_data['pressure'])

    # Finding table data from accessed html file
    table = soup.find("table", attrs={"class": "soundingTable"})
    headings = [th.get_text() for th in table.find("tr").find_all("th")]
    datasets = []
    for row in table.find_all("tr")[1:]:
        dataset = list(zip(headings, (td.get_text()
                                      for td in row.find_all("td"))))
        datasets.append(dataset)

    # Adding each datapoint to dictionary
    for i in range(len(datasets)):
        for j in range(13):
            tuple = datasets[i][j]
            element = list(tuple)
            if element[0] == 'PRES':
                all_data['pressure'].append(float(element[1]))
            elif element[0] == 'HGHT':
                all_data['height'].append(float(element[1]))
            elif element[0] == 'TEMP':
                all_data['temperature'].append(float(element[1]))
            elif element[0] == 'RELH':
                all_data['humidity'].append(float(element[1]))
            elif element[0] == 'DRCT':
                all_data['wind_direction'].append(float(element[1]))
            elif element[0] == 'SKNT':
                all_data['wind_speed'].append(float(element[1]))

    for i in range(len(all_data['pressure'])-1-prevLength):
        all_data['latitude'].append('')
        all_data['longitude'].append('')
