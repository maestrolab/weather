import pickle
from weather.boom import boom_runner, process_data
from geojson import Polygon, Feature, FeatureCollection, dump
from ast import literal_eval

year = '2018'
month = '06'
day = '18'
hours = '12'

data, altitudes = process_data(day, month, year, hours, 0,
                               directory='../../data/weather/')
latlon_list = list(data.keys())

features = []
for i in range(len(data)):
    lat, lon = literal_eval(latlon_list[i])
    hum = data[latlon_list[i]]['humidity']
    polygon = Polygon([[(lon-0.5, lat-0.5),(lon+0.5,lat-0.5),(lon+0.5,lat+0.5),(lon-0.5,lat+0.5),(lon-0.5, lat-0.5)]])
    features.append(Feature(geometry=polygon, properties={"Humidity": hum[-1][-1]}))
# add more features...
# features.append(...)

feature_collection = FeatureCollection(features)

with open('temperature.geojson', 'w') as f:
    dump(feature_collection, f)
