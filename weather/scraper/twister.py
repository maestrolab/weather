import pickle
import copy
import math
import csv
import requests
import numpy as np
import unicodedata as ud
from weather import convert_to_fahrenheit, windToXY, makeFloats
from bs4 import BeautifulSoup


def scraper(YEAR, MONTH, DAY, HOUR, x=np.linspace(13, 58, 46),
            y=np.linspace(-144, -53, 92)):

    # Initialize data dictionary
    all_data = {'latitude': [], 'longitude': [], 'pressure': [], 'height': [],
                'temperature': [], 'humidity': [], 'wind_direction': [],
                'wind_speed': []}

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
                appendToDictionary(latitude, longitude, all_data, soup)

            except ValueError:
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


def process_data(day, month, year, hour, altitude,
                 directory='../data/weather/',
                 outputs_of_interest=['temperature', 'height',
                                      'humidity', 'wind_speed',
                                      'wind_direction', 'pressure',
                                      'latitude', 'longitude'],
                 convert_celcius_to_fahrenheit=False,
                 data=None):
    ''' process_data makes a dictionary output that contains the lists
    specified by the strings given in outputs_of_interest
    '''
    if data is None:
        all_data = pickle.load(open(directory + year +
                                    "_" + month + "_" + day + "_" + hour +
                                    ".p", "rb"))
    else:
        all_data = data
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
    output.pop('wind_speed', None)
    output.pop('wind_direction', None)

    # Prepare for sBOOM
    data = {}
    for key in output.keys():
        lat = output['latitude']
        lon = output['longitude']
        height = output['height']
        if key not in ['latitude', 'longitude', 'height']:
            data, elevations = output_for_sBoom(output[key],
                                               key, altitude, lat,
                                               lon, height, data)
    return data, elevations


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

    elevations = {}
    while i < len(lat):
        if i > 0:
            # appending to mini-list
            if lat[i] == 0:
                temp_height.append(height[i])
                temp_li.append(li[i])
                k += 1
                i += 1
            else:
                # combining height and weather mini lists for storage
                temp_combo_li = combineLatLon(temp_height, temp_li)

                # making sure first two heights aren't the same
                if temp_combo_li[0][0] == temp_combo_li[1][0]:
                    temp_combo_li.pop(0)

                # key is location of previous latlon in big list
                key = '%i, %i' % (lat[i-k], lon[i-k])

                # appending mini-list to dictionary at latlon key
                if d:
                    data[key][keyName] = temp_combo_li
                else:
                    data[key] = {keyName: temp_combo_li}
                elevations[key] = ground_level

                # clearing mini-list and restarting
                temp_height = []
                temp_li = []
                temp_combo_li = []
                k = 0
                temp_height.append(height[i])
                ground_level = temp_height[0]
                ground_altitudes.append(ALT)
                temp_height[0] = temp_height[0]
                temp_li.append(li[i])

                k += 1
                i += 1

        # getting first element in big list
        else:
            temp_height.append(height[i])
            ground_level = temp_height[0]

            ground_altitudes = [ALT]
            temp_height[0] = temp_height[0]
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

    return data, elevations


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
