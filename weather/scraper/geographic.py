import requests
import urllib
import pandas as pd
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

def elevation_function(lat_all,lon_all):
    """Query service using lat, lon. add the elevation values as a new column."""
    url = r'https://nationalmap.gov/epqs/pqs.php?'
    
    df = pd.DataFrame({'lat': lat_all, 'lon': lon_all})
    elevations = []
    for lat, lon in zip(df['lat'], df['lon']):
        # define rest query params        
        params = {
            'output': 'json',
            'x': lon,
            'y': lat,
            'units': 'Meters'
        }
        
        # format query string and return query value
        done = False
        while not done:
            try:
                result = requests.get((url + urllib.parse.urlencode(params)))
                done = True
            except:
                pass
        elevations.append(result.json()['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation'])

    df['elev_meters'] = elevations
    # At sea-level or unknown elevations, it just returns astring. Replace those for zero
    df.loc[df['elev_meters'] == '-1000000'] = 0
    df['elev_feet'] = 3.28084*df['elev_meters']
    df.head()
    return df
    
def get_elevation(lat_all, lon_all):
    elevations = []
    df = pd.DataFrame({'lat': lat_all, 'lon': lon_all})
    for lat, lon in zip(lat_all, lon_all):
        query = ('https://api.open-elevation.com/api/v1/lookup'
                 f'?locations={lat},{lon}')
        print(lat, lon, query)
        r = requests.get(query).json()  # json object, various ways you can extract value
        # one approach is to use pandas json functionality:
        elevation = pd.io.json.json_normalize(r, 'results')['elevation'].values[0]
        elevations.append(elevation)
    df['elevation'] = elevations
    return df