from scipy.io import loadmat
import numpy as np
import copy
from weather import convert_to_fahrenheit


def process(filename):
    data = loadmat(filename, struct_as_record=False)['s'][0][0]
    data.height = data.height[0] * 0.3048  # Convert to meters
    data.temperature = data.temperature[0]
    data.wind_x = data.wind_x[0]
    data.wind_y = data.wind_y[0]
    data.humidity = data.humidity[0]
    data.pressure = data.pressure[0, 0]
    data.lon = data.lon[0]
    data.lat = data.lat[0]
    data.elevation = 10.**5/2.5577*(1-(data.pressure[:, :]/101325)**(1/5.2558)) / 0.3048
    return data


def output_for_sBoom(data, longitude, latitude, aircraft_altitude,
                     convert_K_to_F=True):
    '''output_for_sBoom takes a weather variable list, list keyName, and
    a max altitude (ALT) as user defined inputs. It also requires the
    existance of a dictionary data, and the lat, lon, and height lists
    from the openPickle function. Using these, it makes a dictionary
    with first key being a lat,lon point and second key being the
    name of the weather variable.
    '''

    weather = {}
    try:
        index_lon = np.where(data.lon == longitude)[-1][0]
        index_lat = np.where(data.lat == latitude)[-1][0]
    except(ValueError):
        raise('Longitude, Latitude combination not in database')

    # Consider elevation and round up (because of sboom input) for altitude
    height_above_ground = data.height[:, index_lat, index_lon]
    height_above_ground = np.around(height_above_ground.tolist(), decimals=1)

    # Convert temperature from Kelvin to Farenheight
    temperature = copy.deepcopy(data.temperature[:, index_lat, index_lon])
    if convert_K_to_F:
        temperature = (temperature - 273.15) * 9/5. + 32

    weather['wind'] = np.array([height_above_ground,
                                data.wind_x[:, index_lat, index_lon],
                                data.wind_y[:, index_lat, index_lon]]).T

    weather['temperature'] = np.array([height_above_ground,
                                       temperature]).T
    weather['humidity'] = np.array([height_above_ground,
                                    data.humidity[:, index_lat, index_lon]]).T

    for key in weather:
        weather[key] = weather[key].tolist()
    return([weather['temperature'], weather['wind'], weather['humidity']],
           aircraft_altitude-data.elevation[index_lat, index_lon])
