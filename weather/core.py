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
try:
    from mpl_toolkits.basemap import Basemap
except:
    pass


def convert_to_fahrenheit(temperature_C):
    '''convert_to_fahrenheit takes a list of temperatures in degrees
    celsius and converts the values to units of degrees fahrenheit.
    '''
    if type(temperature_C) == list:
        temperature_C = np.array(temperature_C)
    temperature_F = temperature_C*(9/5)+32

    return temperature_F


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
