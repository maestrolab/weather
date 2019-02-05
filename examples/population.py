from census import Census
from us import states
import numpy as np
from scipy.interpolate import Rbf
import pickle

from aeropy.xfoil_module import output_reader
from weather import process_noise


# Creating surrogate surface from pldb data
day = '18'
month = '06'
year = '2018'
hour = '12'

filename = "../data/noise/" + year + "_" + month + "_" + day + "_" + hour
data = process_noise(filename)
lon, lat, noise = data.T
rbf = Rbf(lon, lat, noise)  # compared for same lat and lon and they look good

# Extracting population data
property = 'B01003_001E'  # Total population
api_key = "d336cbb942af711df388ef67fda11759383df1a0"
c = Census(api_key)

# Extracting location data
type_structure = ['string', 'string', 'string', 'float', 'float',
                  'float', 'float', 'float', 'float']
raw_data = output_reader('../data/us_census/location.txt',
                         type_structure=type_structure)
data = []
undesired_states = ['02', '15']
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

g = open("../data/noise/noise_per_county.p", "wb")
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
print(data)

print(lat)
print(lon)
m.scatter(map_lon, map_lat, marker='D', color='m')
plt.show()
print(data)
