import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pickle
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import copy
import random
from weather import makeFloats


def contour(data, levels=None, label="Perceived Loudness, PLdB",
            index_altitude=None):
    matplotlib.use('TkAgg')
    lon, lat, z = data.T

    numcols, numrows = len(set(lon)), len(set(lat))

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
    try:
        print(map_lon.shape, map_lat.shape, z.shape)
        print(z)
        if index_altitude is None:
            zi = griddata((map_lon, map_lat), z, (xi, yi), method='linear')
        else:
            # z = z.T[index_altitude, ]
            z_altitude = []
            for i in range(len(z)):
                z_altitude.append(z[i][index_altitude][1])
            z_altitude = np.array(z_altitude)
            zi = griddata((map_lon, map_lat), z_altitude, (xi, yi),
                          method='linear')
    except(ValueError):
        print('For weather z is a function of altitude. Choose index' +
              'for altitude (pressure) of interest.')
    plt.scatter(xi, yi)
    # contour plot
    if levels is None:
        m.contourf(xi, yi, zi, cmap=cm.coolwarm)
    else:
        m.contourf(xi, yi, zi, cmap=cm.coolwarm, levels=levels)
    print(min(z_altitude), max(z_altitude))
    # colorbar
    cbar = m.colorbar()
    degree_sign = '\N{DEGREE SIGN}'
    cbar.set_label(label)
    plt.show()
