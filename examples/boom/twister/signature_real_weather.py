import pickle
from weather.boom import boom_runner, prepare_weather_sBoom
from weather.scraper.twister import process_data

day = '18'
month = '06'
year = '2018'
hour = '12'
lat = 32
lon = -100
alt = 50000.

data, elevations = process_data(day, month, year, hour, alt,
                               directory='../../../data/weather/twister/',
                               convert_celcius_to_fahrenheit=True)

key = '%i, %i' % (lat, lon)
weather_data = data[key]

# Height to ground (HAG)
index = list(data.keys()).index(key)
elevation = elevations[key]

sBoom_data = prepare_weather_sBoom(data, index)
[temperature, wind, humidity] = sBoom_data
noise = boom_runner(sBoom_data, alt, elevation)

print(noise)
