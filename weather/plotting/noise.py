#!python3
'''
Makes a contour plot of noise values output from pyLdB.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pickle
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import copy
import random
from weather import makeFloats

def contour(filename, levels=None, transformation=None, label="Perceived Loudness, PLdB"):
    noise_data = pickle.load(open(filename + '.p', 'rb'))

    lat = []
    lon = []
    latlon = copy.deepcopy(noise_data['latlon'])

    for i in range(len(latlon)):
        latlon_temp = [int(s) for s in latlon[i].split(',')]
        lat.append(latlon_temp[0])
        lon.append(latlon_temp[1])

    numcols, numrows = len(lon), len(lat)


    # REMOVE this when z should be pyLdB input
    z = copy.deepcopy(noise_data['noise'])
    if transformation is not None:
        z = transformation(z)
    # Make lists into arrays to graph
    lon = makeFloats(lon)
    lat = makeFloats(lat)
    lon = np.array(lon)
    lat = np.array(lat)

    fig = plt.figure(figsize=(12, 6))

    # bounds = np.arange(0,110,10) - FIXME to match output
    m = Basemap(projection='merc', llcrnrlat=13, urcrnrlat=58,
                llcrnrlon=-144, urcrnrlon=-53, resolution='c')
    map_lon, map_lat = m(*(lon, lat))

    m.drawstates()
    m.drawcountries(linewidth=1.0)
    m.drawcoastlines()

    # target grid to interpolate to
    xi = np.linspace(map_lon.min(), map_lon.max(), numcols)
    yi = np.linspace(map_lat.min(), map_lat.max(), numrows)
    xi, yi = np.meshgrid(xi, yi)

    # interpolate
    print(len(z))
    zi = griddata((map_lon, map_lat), z, (xi, yi), method='linear')

    # contour plot
    if levels is None:
        m.contourf(xi, yi, zi, cmap=cm.coolwarm)
    else:
        m.contourf(xi, yi, zi, cmap=cm.coolwarm, levels=levels)

    # colorbar
    cbar = m.colorbar()
    degree_sign = '\N{DEGREE SIGN}'
    cbar.set_label(label)

    plt.show()
