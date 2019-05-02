import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle

from weather.scraper.noaa import process
year = '2018'
month = '06'
day = '18'
hour = '12'
alt_ft = 50000
directory = '../../../data/noise/'
filename = directory + year + month + day + '_' + hour + '_' + \
    str(alt_ft) + ".p"

f = open(filename, "rb")
data = pickle.load(f)

fig = plt.figure(figsize=(12, 6))
from mpl_toolkits.basemap import Basemap
# bounds = np.arange(0,110,10) - FIXME to match output
m = Basemap(projection='merc', llcrnrlat=13, urcrnrlat=58,
            llcrnrlon=-144, urcrnrlon=-53, resolution='c')
map_lon, map_lat = m(*(data.lon_grid, data.lat_grid))
m.drawstates()
m.drawcountries(linewidth=1.0)
m.drawcoastlines()

plt.contourf(map_lon, map_lat, np.array(data.noise).reshape(data.lon_grid.shape),
             cmap=cm.coolwarm)

cbar = m.colorbar()
degree_sign = '\N{DEGREE SIGN}'
label = "Relative Humidity"
cbar.set_label(label)
plt.show()
