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
