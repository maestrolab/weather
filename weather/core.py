#!python3
'''
Code that consolidates all aux needed to run any file in
 weather_module repository in alphabetical order.
'''

import requests
from bs4 import BeautifulSoup
import unicodedata as ud
import datetime
import pickle
import copy
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import io
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from mpl_toolkits.basemap import Basemap


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


def windToXY(sknt, drct):
    ''' windToXY takes wind speed in knots and wind direction in degrees
    clockwise from North lists and converts them to wind velocities in
    m/s along the x (East) and y (North) axes
    '''

    # conversion to m/s
    wind_speed = np.array([x*0.51444 for x in sknt])
    # converting degrees to radians
    wind_direction = np.array([math.radians(x) for x in drct])

    # directions are from North, clockwise so x is sin(drct)
    wind_x = wind_speed*np.sin(wind_direction)
    wind_y = wind_speed*np.cos(wind_direction)

    return wind_x, wind_y


def makeFloats(w_var):
    '''makeFloats takes a weather variable as an input list and makes
    every element in the list a float for use in mathematical
    calculations. This function also converts any '' in the list to a 0.
    '''
    for i in range(len(w_var)):
        if w_var[i] == '':
            w_var[i] = 0

        w_var[i] = float(w_var[i])

    return w_var
