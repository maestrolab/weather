import pickle
from weather.boom import boom_runner, process_data
from geojson import Point, Feature, FeatureCollection, dump
from ast import literal_eval

year = '2018'
month = '06'
day = '18'
hours = '12'

data, altitudes = process_data(day, month, year, hours, 0,
                              directory='../data/weather/')
latlon_list = list(data.keys())

features = []
for i in range(len(data)):
    lat, lon = literal_eval(latlon_list[i])
    temp = data[latlon_list[i]]['temperature']
    point = Point((lon, lat))
    features.append(Feature(geometry=point, properties={"Temperature": temp[-1][-1]}))

# add more features...
# features.append(...)

feature_collection = FeatureCollection(features)

with open('myfile1.geojson', 'w') as f:
    dump(feature_collection, f)
