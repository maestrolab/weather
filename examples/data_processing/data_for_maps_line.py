from weather.filehandling import output_reader
from geojson import LineString, Feature, FeatureCollection, dump

type_structure = ['string', 'string', 'float', 'float', 'float', 'float']

path = 'MMMX-CYQB'

data = output_reader('../../data/flight_plan/' + path + '.csv', ',',
              header = ['Dumb', 'Dumber', 'altitude', 'latitude', 'longitude','distance'],
              type_structure = type_structure)
print(data.keys())
features = []
points = []
for i in range(len(data['latitude'])):
    lat = data['latitude'][i]
    lon = data['longitude'][i]
    points.append((lon,lat),)

line = LineString(points)
features.append(Feature(geometry=line))

feature_collection = FeatureCollection(features)


with open('flightPath' + path + '.geojson', 'w') as f:
    dump(feature_collection, f)
