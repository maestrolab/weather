import matplotlib.pyplot as plt
import numpy as np
import pickle

# Both functions included as they have not been added to weather library tools
def package_data(data1, data2=None, method='unpack'):
    '''package_data packs or unpacks data in the form [[data1, data2]]'''
    if method == 'pack':
        packed_data = [[data1[i],data2[i]] for i in range(len(data1))]
        return packed_data
    elif method == 'unpack':
        unpacked_data_1 = [d[0] for d in data1]
        unpacked_data_2 = [d[1] for d in data1]
        return unpacked_data_1, unpacked_data_2

def convert_to_celcius(temperature_F):
    if type(temperature_F) == list:
        temperature_F = np.array(temperature_F)
    temperature_C = (5/9)*(temperature_F-32)
    return temperature_C

def calculate_vapor_pressures(humidities, temperatures):
    '''calculate_vapor_pressures calculates the saturation_vapor_pressures
    and the actual_vapor_pressures given a temperature and humidity
    profile (equations from Arden Buck (1981))
    '''
    saturation_vps = []
    for t in temperatures:
       if t>=0:
           saturation_vps.append(0.61121*np.exp((18.678-(t/234.5))*(t/(257.14+t))))
       else:
           saturation_vps.append(0.61115*np.exp((23.036-(t/333.7))*(t/(279.82+t))))
    actual_vps = [humidities[i]/100*saturation_vps[i] for i in range(len(humidities))]
    return actual_vps

def prepare_vapor_pressures(humidity_profiles, temperature_profiles):
    '''prepare_vapor_pressures calculates the vapor pressure profile for a set
    of humidity profiles
    '''
    vapor_pressure_profiles = []
    for i in range(len(temperature_profiles)):
        temps = temperature_profiles[i]
        rhs = humidity_profiles[i]
        altitudes, relative_humidities = package_data(rhs)
        altitudes_, temperatures = package_data(temps)
        vapor_pressures = calculate_vapor_pressures(relative_humidities, temperatures)
        vapor_pressure_profile = package_data(altitudes, vapor_pressures, method='pack')
        vapor_pressure_profiles.append(vapor_pressure_profile)

    return vapor_pressure_profiles

YEAR = '2018'
MONTH = '06'
DAY = '18'
HOUR = '00'
altitude = 50000
directory = './'
locations = ['72469']  # Corresponds to Fort Worth/Dallas

f = open(locations[0] + '.p', 'rb')
data = pickle.load(f)
f.close()

for i in range(len(data['temperature'])):
    alts, temps = package_data(data['temperature'][i])
    temps_c = convert_to_celcius(temps)
    data['temperature'][i] = package_data(alts, list(temps_c), method='pack')

data['vapor_pressures'] = prepare_vapor_pressures(data['humidity'],
                                                  data['temperature'])

plt.figure()
for i in range(len(np.array(data['noise']))):
    alt, property = np.array(data['vapor_pressures'][i]).T
    plt.plot(property, alt, 'k', alpha=0.05)
plt.ylabel('Altitude (m)')
plt.xlabel('Vapor Pressure (kPa)')

plt.show()
