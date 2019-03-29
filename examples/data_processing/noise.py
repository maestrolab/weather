import pickle
from weather.boom import boom_runner, process_data

day = '18'
month = '06'
year = '2018'
hour = '12'
alt_ft = 45000.
alt = alt_ft * 0.3048

data, altitudes = process_data(day, month, year, hour, alt,
                               directory='../data/weather/')

noise_data = {'latlon': [], 'noise': []}

index = 2337  # index for worst case scenario
noise_data['latlon'].append(list(data.keys())[index])
noise_data['noise'].append(boom_runner(data, altitudes[index], index))

print(noise_data['latlon'], noise_data['noise'])
