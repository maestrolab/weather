import pickle
from weather.boom import boom_runner, process_data
from geojson import Polygon, Feature, FeatureCollection, dump
from ast import literal_eval

def create_geojson(property):
    features = []
    for i in range(len(data)):
        lat, lon = literal_eval(latlon_list[i])
        prop = data[latlon_list[i]][property]
        polygon = Polygon([[(lon-0.5, lat-0.5),(lon+0.5,lat-0.5),
                            (lon+0.5,lat+0.5),(lon-0.5,lat+0.5),
                            (lon-0.5, lat-0.5)]])
        features.append(Feature(geometry=polygon, properties={property: prop[0][-1]}))
        print(prop)

    feature_collection = FeatureCollection(features)

    with open(str(property) + '.geojson', 'w') as f:
        dump(feature_collection, f)

year = '2018'
month = '06'
day = '18'
hours = '12'

data, altitudes = process_data(day, month, year, hours, 0,
                               directory='../../data/weather/')
latlon_list = list(data.keys())

properties = ['temperature', 'humidity']
for property in properties:
    create_geojson(property)
