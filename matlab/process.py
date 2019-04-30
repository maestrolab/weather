import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

year = '2018'
month = '06'
day = '18'
hour = '12'
filename = year + month + day + '_' + hour + '.mat'

#
data = loadmat(filename, struct_as_record=False)['s'][0][0]
data.height = data.height[0]
data.temperature = data.temperature[0]
data.wind_x = data.wind_x[0]
data.wind_y = data.wind_y[0]
data.humidity = data.humidity[0]


fig = plt.figure(figsize=(12, 6))

from mpl_toolkits.basemap import Basemap
# bounds = np.arange(0,110,10) - FIXME to match output
m = Basemap(projection='merc', llcrnrlat=13, urcrnrlat=58,
            llcrnrlon=-144, urcrnrlon=-53, resolution='c')
map_lon, map_lat = m(*(data.lon, data.lat))
print(map_lon)
m.drawstates()
m.drawcountries(linewidth=1.0)
m.drawcoastlines()

print(type(data.lon), type(data.lat), type(data.humidity[0, :, :]))
print(data.lon.shape, data.lat.shape, data.humidity.shape)

plt.contourf(map_lon, map_lat, data.humidity[0, :, :])
cbar = m.colorbar()
degree_sign = '\N{DEGREE SIGN}'
label = "Humidity"
cbar.set_label(label)
plt.show()
