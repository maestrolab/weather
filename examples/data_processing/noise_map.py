import pickle
from weather import process_noise
from geojson import Polygon, Feature, FeatureCollection, dump

day = '18'
month = '06'
year = '2018'
hour = '12'

filename = "../../data/noise/" + year + month + day + '/full'
data = process_noise(filename)

features = []
for i in range(len(data)):
    print(i)
    lat, lon, noise = data[i]
    polygon = Polygon([[(lat-0.5, lon-0.5),(lat+0.5,lon-0.5),
                        (lat+0.5,lon+0.5),(lat-0.5,lon+0.5),
                        (lat-0.5, lon-0.5)]])
    features.append(Feature(geometry=polygon, properties={"Noise": noise}))

# add more features...
# features.append(...)

feature_collection = FeatureCollection(features)

with open('noise.geojson', 'w') as f:
    dump(feature_collection, f)
