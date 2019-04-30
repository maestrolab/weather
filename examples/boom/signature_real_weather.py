import pickle
from weather.boom import boom_runner, process_data

day = '18'
month = '06'
year = '2018'
hour = '12'
lat = 32
lon = -100
alt_ft = 45000.
alt = alt_ft * 0.3048

data, altitudes = process_data(day, month, year, hour, alt,
                               directory='../../data/weather/',
                               convert_to_fahrenheit=True)

key = '%i, %i' % (lat, lon)
weather_data = data[key]

# Height to ground (HAG)
index = list(data.keys()).index(key)
height_to_ground = altitudes[index]  # In meters

noise = boom_runner(data, height_to_ground, index)

print(noise)
