import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate
import pickle

from weather.scraper.noaa import process
year = '2018'
month = '06'
day = '21'
hour = '12'
alt_ft = 50000
directory = '../../../data/noise/'
filename = directory + year + month + day + '_' + hour + '_' + \
    str(alt_ft) + ".p"
# filename = directory + 'standard_' + \
#     str(alt_ft) + ".p"
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

# Get rid of NaN
array = np.ma.masked_invalid(np.array(data.noise).reshape(data.lon_grid.shape))

#get only the valid values
lon1 = data.lon_grid[~array.mask]
lat1 = data.lat_grid[~array.mask]
newarr = array[~array.mask]

noise = interpolate.griddata((lon1, lat1), newarr.ravel(),
                          (data.lon_grid, data.lat_grid),
                             method='cubic')
# Plotting                       
baseline = np.min(noise)
print(baseline, np.max(noise) - np.min(noise))
plt.contourf(map_lon, map_lat, np.array(noise).reshape(data.lon_grid.shape),
             cmap=cm.coolwarm, levels=np.linspace(64, 80, 17))

cbar = m.colorbar()
degree_sign = '\N{DEGREE SIGN}'
label = "Perceived level in dB (PLdB)"
cbar.set_label(label)
plt.show()
