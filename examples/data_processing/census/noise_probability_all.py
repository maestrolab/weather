from census import Census
from us import states
import numpy as np
from scipy import interpolate
import pickle

from aeropy.xfoil_module import output_reader

# Creating surrogate surface from pldb data
year = '2018'
month = '06'
day = '21'
hour = '00'
altitude = '50000'
input_directory = '../../../data/noise/'
output_directory = '../../../data/noise/county/'
filename =  year + month + day + '_' + hour + '_' + altitude + '.p'

alt_ft = 50000

# Process weather data
f = open(input_directory +filename, 'rb')
data = pickle.load(f)
f.close()

lon = data.lonlat[:, 0]
lat = data.lonlat[:, 1]

# Get rid of NaN
array = np.ma.masked_invalid(np.array(data.noise).reshape(data.lon_grid.shape))

#get only the valid values
lon1 = data.lon_grid[~array.mask]
lat1 = data.lat_grid[~array.mask]
newarr = array[~array.mask]

noise = interpolate.griddata((lon1, lat1), newarr.ravel(),
                          (data.lon_grid, data.lat_grid),
                             method='cubic')
print(np.shape(lon))
print(np.shape(lat))
print(np.shape(noise))
rbf = interpolate.Rbf(lon, lat, noise)  # compared for same lat and lon and they look good

# Extracting population data
property = 'B01003_001E'  # Total population
api_key = "d336cbb942af711df388ef67fda11759383df1a0"
c = Census(api_key)

# Extracting location data
type_structure = ['string', 'string', 'string', 'float', 'float',
                  'float', 'float', 'float', 'float']
raw_data = output_reader('../../data/us_census/location.txt',
                         type_structure=type_structure)
data = []
undesired_states = ['02', '15']
print(len(raw_data['GEOID']))
for i in range(len(raw_data['GEOID'])):
    print(i)
    state_id = raw_data['GEOID'][i][:2]
    county_id = raw_data['GEOID'][i][2:5]
    if raw_data['GEOID'][i][:2] not in undesired_states:
        lon = float(raw_data['LON'][i])
        lat = float(raw_data['LAT'][i])
        pop = c.acs5.state_county(property, state_id, county_id)[0][property]
        noise = rbf(lon, lat)
        data.append([lon, lat, pop, noise])
data = np.array(data)

g = open(output_directory + filename, "wb")
pickle.dump(data, g)
g.close()

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

fig = plt.figure(figsize=(12, 6))

# bounds = np.arange(0,110,10) - FIXME to match output
lon, lat, pop, noise = data.T
m = Basemap(projection='merc', llcrnrlat=13, urcrnrlat=58,
            llcrnrlon=-144, urcrnrlon=-53, resolution='c')
map_lon, map_lat = m(*(lon, lat))

m.drawstates()
m.drawcountries(linewidth=1.0)
m.drawcoastlines()

m.scatter(map_lon, map_lat, marker='D', color='m')
plt.show()
print(data)