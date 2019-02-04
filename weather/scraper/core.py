#!python3
'''
Code that consolidates all aux needed to run any file in
 weather_module repository in alphabetical order.
'''
import weather

import requests
from bs4 import BeautifulSoup
import unicodedata as ud
import datetime
import pickle
import copy
import math
import csv
import numpy as np


def balloon_scraper(YEAR, MONTH, DAY, HOUR):

    # INPUT YEAR AND MONTH
    YEAR = '2014'
    MONTH = '01'

    # INPUT BEGIN AND END TIME (FORMAT IS DAY/HOUR)
    DAY = '07'
    FROM = DAY + HOUR
    TO = DAY + HOUR

    # LIST OF LOCATIONS ACROSS NA
    locations = ['03953', '04220', '04270', '04360', '08508', '70133', '70200',
                 '70219', '70231', '70261', '70273', '70316', '70326', '70350',
                 '70398', '70414', '71043', '71081', '71109', '71119', '71126',
                 '71600', '71603', '71722', '71811', '71815', '71816', '71823',
                 '71836', '71845', '71867', '71906', '71907', '71908', '71909',
                 '71913', '71924', '71925', '71926', '71934', '71945', '71957',
                 '71964', '72201', '72202', '72206', '72208', '72210', '72214',
                 '72215', '72230', '72233', '72235', '72240', '72248', '72249',
                 '72251', '72261', '72265', '72274', '72293', '72305', '72317',
                 '72318', '72327', '72340', '72357', '72363', '72364', '72365',
                 '72376', '72388', '72393', '72402', '72403', '72426', '72440',
                 '72451', '72456', '72469', '72476', '72489', '72493', '72501',
                 '72518', '72520', '72528', '72558', '72562', '72572', '72582',
                 '72597', '72632', '72634', '72645', '72649', '72659', '72662',
                 '72672', '72681', '72694', '72712', '72747', '72764', '72768',
                 '72776', '72786', '72797', '74005', '74389', '74455', '74494',
                 '74560', '74646', '74794', '76256', '76394', '76458', '76526',
                 '76595', '76612', '76644', '76654', '76679', '76805', '78016',
                 '78073', '78384', '78397', '78486', '78526', '78583', '78807',
                 '78897', '78954', '78970', '91285', '80222', '82022', '91165',
                 '91285']

    f = open('WBData.csv', 'w')
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
        print(counter_x)

        page = requests.get(page)
        page
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
                        counter_filter += 1
                        if counter_filter == 8:
                            counter_filter = 0
                        if counter_filter == 0:
                            f.write(str(latitude) + ',' + str(longitude) +
                                    ',' + soup[i]+'\n')

    f.close()

    f = open("WB" + YEAR + "_" + MONTH + "_" + DAY + ".csv", "w")
    f.write('Latitude' + ',' + 'Longtitude' + ',' + 'Pressure [hPa]' +
            ',' + 'Height [m]' + ',' + 'Temperature [C]' + ',' +
            'Relative Humidity [%]' + ',' + 'Wind Direction [deg]' +
            ',' + 'Wind Speed [knot]' + '\n')
    f.close()

    with open("WBData.csv", "r") as source:
        rdr = csv.reader(source)
        with open("WB" + YEAR + "_" + MONTH + "_" + DAY + ".csv", "a") as result:
            wtr = csv.writer(result)
            for r in rdr:
                wtr.writerow((r[0], r[1], r[3], r[2], r[4], r[6], r[8], r[9]))
                all_data['latitude'].append(float(r[0]))
                all_data['longitude'].append(float(r[1]))
                all_data['pressure'].append(float(r[3]))
                all_data['height'].append(float(r[2]))
                all_data['temperature'].append(float(r[4]))
                all_data['humidity'].append(float(r[6]))
                all_data['wind_direction'].append(float(r[8]))
                all_data['wind_speed'].append(float(r[9]))

    return all_data


def noaa_scraper(YEAR, MONTH, DAY, HOUR):

    # Initialize data dictionary
    all_data = {'latitude': [], 'longitude': [], 'pressure': [], 'height': [],
                'temperature': [], 'humidity': [], 'wind_direction': [],
                'wind_speed': []}

    # Lat, Lon Locations on TwisterData.com grid
    x = np.linspace(13, 14, 1)  # 58, 46)  # lat - (13,58)
    y = np.linspace(-144, -53, 92)  # lon - (-144,-53)

    for j in range(len(x)):
        for k in range(len(y)):
            X = str(x[j])
            Y = str(y[k])
            print(X, Y)
            # Access Website
            q = 0
            while q < 1:
                try:
                    r = requests.get('http://www.twisterdata.com/index.php?' +
                                     'sounding.lat=' + X + '&sounding.lon=' + Y +
                                     '&prog=forecast&model=GFS&grid=3&model_' +
                                     'yyyy=' + YEAR + '&model_mm=' + MONTH +
                                     '&model_dd=' + DAY + '&model_init_hh=' +
                                     HOUR + '&fhour=00' + '&parameter=TMPF&level' +
                                     '=2&unit=M_ABOVE_GROUND&maximize=n&mode=sin' +
                                     'glemap&sounding=y&output=text&view=large&a' +
                                     'rchive=false&sndclick=y', timeout=5)
                    q += 1
                except IOError:
                    print('ERROR')
                    q = 0

            html = r.text
            soup = BeautifulSoup(html, "html.parser")
            try:
                # Finding Latitude and Longitude for each accessed webpage
                lat = soup.find("div", attrs={"class": "soundingLatLonHeader"})
                label = lat.get_text()
                words = ud.normalize('NFKD', label).encode('ascii', 'ignore')
                word_list = words.split()
                latitude = word_list[12]
                long_1 = word_list[15]
                longitude = long_1[:-52]

                # check if latitude and longitude are a numbers
                latnum = latitude
                latnum = float(latnum)
                longnum = longitude
                longnum = float(longnum)

                # add elements to dictionary
                weather.appendToDictionary(latitude, longitude, all_data, soup)

            except ValueError:
                doNothingVariable = 0
                print('Value Error - Invalid Date')
                # do nothing
    store(all_data, YEAR, MONTH, DAY, HOUR)


def store(all_data, YEAR, MONTH, DAY, HOUR):
    # making colums to put into the csv file
    rows = zip(all_data['latitude'], all_data['longitude'],
               all_data['pressure'], all_data['height'],
               all_data['temperature'], all_data['humidity'],
               all_data['wind_direction'], all_data['wind_speed'])

    # initializing csv file
    f = open("YEAR" + "_" + MONTH + "_" + DAY + "_" + HOUR + ".csv", "w")
    f.write('Latitude' + ',' + 'Longtitude' + ',' + 'Pressure [hPa]' + ',' +
            'Height [m]' + ',' + 'Temperature [C]' + ',' +
            'Relative Humidity [%]' + ',' + 'Wind Direction [deg]' + ',' +
            'Wind Speed [knot]' + '\n')
    f.close()

    # adding data to csv file in table format
    with open(YEAR + "_" + MONTH + "_" + DAY + "_" + HOUR + ".csv",
              "a") as f:
        wtr = csv.writer(f)
        for row in rows:
            wtr.writerow(row)

    # creating pickle file for later use
    g = open(YEAR + "_" + MONTH + "_" + DAY + "_" + HOUR + ".p", "wb")
    pickle.dump(all_data, g)
    g.close()
