import pickle
from weather import process_noise
from weather.boom import boom_runner, process_data
from geojson import Point, Feature, FeatureCollection, dump

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
    point = Point((lat, lon))
    features.append(Feature(geometry=point, properties={"Noise": noise}))

# add more features...
# features.append(...)

feature_collection = FeatureCollection(features)

with open('noise.geojson', 'w') as f:
    dump(feature_collection, f)
