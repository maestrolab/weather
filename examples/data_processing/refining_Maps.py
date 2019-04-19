import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
import pickle
from weather import process_noise
from geojson import Polygon, Feature, FeatureCollection, dump

def refine_data(data, refinement=1., output_as_meshgrid=False):
    latitude = []
    longitude = []
    noise = []
    for i in range(len(data)):
        lat_i, lon_i, noise_i = data[i]
        latitude.append(lat_i)
        longitude.append(lon_i)
        noise.append(noise_i)

    n_rows = len(set(latitude))
    n_columns = len(set(longitude))
    min_lat, max_lat = min(latitude), max(latitude)
    min_lon, max_lon = min(longitude), max(longitude)

    latitude = np.array(latitude).reshape((n_rows, n_columns))
    longitude = np.array(longitude).reshape((n_rows, n_columns))
    noise = np.array(noise).reshape((n_rows, n_columns))
    # Train Response surface
    rbf = Rbf(latitude, longitude, noise)

    # Grid to interpolate
    x_interpolate = np.linspace(min_lat, max_lat, refinement*n_rows)
    y_interpolate = np.linspace(min_lon, max_lon, refinement*n_columns)
    X,Y = np.meshgrid(x_interpolate, y_interpolate)
    F_interpolate = rbf(X, Y)
    if not output_as_meshgrid:
        return(np.array([X.ravel(), Y.ravel(), F_interpolate.ravel()]).T)
    else:
        return(X, Y, F_interpolate)

def generate_geo(data, refinement = 1.):
    features = []
    s = 0.5/refinement
    for i in range(len(data)):
        print(i)
        lat, lon, noise = data[i]
        polygon = Polygon([[(lat - s, lon - s), (lat + s, lon - s),
                            (lat + s, lon + s), (lat - s, lon + s),
                            (lat - s, lon - s)]])
        features.append(Feature(geometry=polygon, properties={"Noise": noise}))

    # add more features...
    # features.append(...)

    feature_collection = FeatureCollection(features)

    with open('loudness.geojson', 'w') as f:
        dump(feature_collection, f)

day = '18'
month = '06'
year = '2018'
hour = '12'

filename = "../../data/noise/" + year + month + day + '/full'
data = process_noise(filename)

refinement = 1.5
data = refine_data(data, refinement)
generate_geo(data, refinement)
# print(F_interpolate)
# plt.figure()
# # plt.scatter(X, Y, label='Sampled')
# plt.contour(X, Y, F_interpolate)
# # plt.plot(x_interpolate, f_interpolate, label='Interpolated')
# # plt.plot(x_interpolate, x_interpolate**2, label='Real')
# plt.legend()
# plt.show()
