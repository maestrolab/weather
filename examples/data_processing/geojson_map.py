import pickle
import numpy as np
from scipy.interpolate import Rbf
import dask.array as da
from weather.boom import process_data
from geojson import Polygon, Feature, FeatureCollection, dump
from ast import literal_eval

def refine_data(data, property, refinement=1., output_as_meshgrid=False):
    latitude = []
    longitude = []
    profile = []
    for i in range(len(data)):
        lat_i, lon_i = literal_eval(latlon_list[i])
        latitude.append(lat_i)
        longitude.append(lon_i)
        profile.append(data[latlon_list[i]][property][index][-1])

    n_rows = len(set(latitude))
    n_columns = len(set(longitude))
    min_lat, max_lat = min(latitude), max(latitude)
    min_lon, max_lon = min(longitude), max(longitude)

    latitude = np.array(latitude).reshape((n_rows, n_columns))
    longitude = np.array(longitude).reshape((n_rows, n_columns))
    noise = np.array(profile).reshape((n_rows, n_columns))
    # Train Response surface
    rbf = Rbf(latitude, longitude, noise)

    # Grid to interpolate
    x_interpolate = np.linspace(min_lat, max_lat, refinement*n_rows)
    y_interpolate = np.linspace(min_lon, max_lon, refinement*n_columns)
    X,Y = np.meshgrid(x_interpolate, y_interpolate)
    n1 = X.shape[1]
    ix = da.from_array(X, chunks=(1, n1))
    iy = da.from_array(Y, chunks=(1, n1))
    iz = da.map_blocks(rbf, ix, iy)
    F_interpolate = iz.compute()
    if not output_as_meshgrid:
        return(np.array([X.ravel(), Y.ravel(), F_interpolate.ravel()]).T)
    else:
        return(X, Y, F_interpolate)

def generate_geo(data, property, refinement = 1.):
    features = []
    s = 0.5/refinement
    for i in range(len(data)):
        lat, lon, noise = data[i]
        polygon = Polygon([[(lat - s, lon - s), (lat + s, lon - s),
                            (lat + s, lon + s), (lat - s, lon + s),
                            (lat - s, lon - s)]])
        features.append(Feature(geometry=polygon, properties={"Noise": noise}))

    # add more features...
    # features.append(...)

    feature_collection = FeatureCollection(features)

    with open(property + '_' + str(index) + '.geojson', 'w') as f:
        dump(feature_collection, f)

year = '2018'
month = '06'
day = '18'
hours = '12'
index = 11
print('Processing data')
data, altitudes = process_data(day, month, year, hours, 0,
                               directory="../../data/weather/")
latlon_list = list(data.keys())

refinement = 4
properties = ['temperature', 'humidity']
for property in properties:
    print('Refining for ' + property)
    data_i = refine_data(data, property, refinement)
    print('Generate GeoJson for ' + property)
    generate_geo(data_i, property, refinement)
print('All done!')
