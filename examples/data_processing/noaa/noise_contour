import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import cm

from weather.scraper.noaa import process
year = '2018'
month = '06'
day = '18'
hour = '12'
directory = '../../../matlab/'
filename = directory + year + month + day + '_' + hour + '.mat'

index_altitude = 0
#
data = process(filename)
print(data.lonlat.shape)
fig = plt.figure(figsize=(12, 6))
from mpl_toolkits.basemap import Basemap
# bounds = np.arange(0,110,10) - FIXME to match output
m = Basemap(projection='merc', llcrnrlat=13, urcrnrlat=58,
            llcrnrlon=-144, urcrnrlon=-53, resolution='c')
map_lon, map_lat = m(*(data.lon_grid, data.lat_grid))
m.drawstates()
m.drawcountries(linewidth=1.0)
m.drawcoastlines()
print(min(data.humidity[index_altitude, :, :].ravel()),
      max(data.humidity[index_altitude, :, :].ravel()))
plt.contourf(map_lon, map_lat, data.elevation[:, :],
             cmap=cm.coolwarm)

cbar = m.colorbar()
degree_sign = '\N{DEGREE SIGN}'
label = "Relative Humidity"
cbar.set_label(label)
plt.show()
