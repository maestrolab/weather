import pickle
from weather.boom import boom_runner, process_data

day = '18'
month = '06'
year = '2018'
hour = '12_'

alt_ft = 50000.
alt = alt_ft * 0.3048

path = '../../data/'
data, altitudes = process_data(day, month, year, hour, alt,
                               directory=path + 'weather/',
                               convert_to_fahrenheit=True)

key_list = list(data.keys())
for index in range(len(data.keys())):
    key = key_list[index]
    weather_data = data[key]
    # Height to ground (HAG)
    index = list(data.keys()).index(key)
    height_to_ground = altitudes[index]  # In meters

    noise = boom_runner(data, height_to_ground, index)
    print(index, key, noise)
    data[key]['noise'] = noise
    if index % 100 == 99:
        print(index)
        f = open(path + "noise/" + year + month + day + '/' +
                 str(int(round(index, -1))) + ".p", "wb")
        pickle.dump(data, f)
        f.close()
f = open(path + "/noise/" + year + month + day + '_' + hour + '_'
         + str(alt_ft) + ".p", "wb")
pickle.dump(data, f)
