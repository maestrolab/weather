import copy
import pickle
import numpy as np
from scipy import interpolate

try:
    from pyldb import perceivedloudness
    from rapidboom.sboomwrapper import SboomWrapper
except:
    print('Boom analysis is disabled')
from weather import makeFloats, windToXY


def boom_runner(data, cruise_altitude, j,
                nearfield_file='../../data/nearfield/25D_M16_50000ft.p'):
    '''
    Runs sBOOM
     Python3 Version
    '''

    # Define parameters
    ALT_ft = cruise_altitude / 0.3048

    CASE_DIR = "."  # folder where all case files for the tools will be stored
    REF_LENGTH = 32.92
    MACH = 1.6
    R_over_L = 1

    # Define latitude and longitude
    key = list(data.keys())[j]

    # get pressure signature from pickle
    nearfield_sig = pickle.load(open(nearfield_file, "rb"))

    # initialize sBOOM
    sboom = SboomWrapper(CASE_DIR, exe="sboom_windows.dat.allow")

    # temperature input (altitude ft, temperature F)
    temperature = data[key]['temperature']

    # wind input (altitude ft, wind X, wind Y)
    wind = []
    wind = data[key]['wind_x']  # data[key]['wind_y']]
    for i in range(len(wind)):
        wind[i].append(data[key]['wind_y'][i][1])

    # change mach_number for each iteration based on wind
    mach = MACH  # MachModifier(DIRECTION, MACH, ALT, wind)

    # wind input (altitude ft, humidity %)
    humidity = data[key]['humidity']

    # update sBOOM settings and run
    sboom.set(mach_number=mach,
              altitude=ALT_ft,
              propagation_start=R_over_L*REF_LENGTH*3.28084,
              altitude_stop=0.,
              output_format=0,
              input_xdim=2,
              signature=nearfield_sig,
              input_temp=temperature,
              input_wind=wind,
              input_humidity=humidity)

    sboom_results = sboom.run()

    ground_sig = sboom_results["signal_0"]["ground_sig"]

    # grab the loudness level
    noise_level = perceivedloudness(ground_sig[:, 0], ground_sig[:, 1])

    return noise_level


def process_data(day, month, year, hour, altitude,
                 directory='../data/weather/',
                 outputs_of_interest=['temperature', 'height',
                                      'humidity', 'wind_speed',
                                      'wind_direction', 'pressure',
                                      'latitude', 'longitude']):
    ''' process_data makes a dictionary output that contains the lists
    specified by the strings given in outputs_of_interest
    '''

    all_data = pickle.load(open(directory + year +
                                "_" + month + "_" + day + "_" + hour +
                                ".p", "rb"))

    # Reading data for selected properties
    if outputs_of_interest == 'all':
        output = all_data
    else:
        output = {}

        for key in outputs_of_interest:
            output[key] = copy.deepcopy(all_data[key])

    # Make everything floats
    for key in outputs_of_interest:
        output[key] = makeFloats(output[key])

    # Convert wind data
    wind_x, wind_y = windToXY(output['wind_speed'],
                              output['wind_direction'])
    output['wind_x'] = wind_x
    output['wind_y'] = wind_y
    output.pop('wind_speed', None)
    output.pop('wind_direction', None)

    # Prepare for sBOOM
    data = {}
    for key in output.keys():
        lat = output['latitude']
        lon = output['longitude']
        height = output['height']
        if key not in ['latitude', 'longitude', 'height']:
            data, ground_altitudes = output_for_sBoom(output[key],
                                                      key, altitude, lat,
                                                      lon, height, data)
    return data, ground_altitudes


def output_for_sBoom(li, keyName, ALT, lat, lon, height, data):
    '''output_for_sBoom takes a weather variable list, list keyName, and
    a max altitude (ALT) as user defined inputs. It also requires the
    existance of a dictionary data, and the lat, lon, and height lists
    from the openPickle function. Using these, it makes a dictionary
    with first key being a lat,lon point and second key being the
    name of the weather variable.
    '''
    temp_height = []
    temp_li = []
    temp_combo_li = []
    d = copy.deepcopy(data)
    k = 0
    i = 0
    ground_level = 0
    ground_altitudes = []
    # ground_level = 0
    # ground_altitudes = []
    while i < len(lat):
        if i > 0:
            # appending to mini-list
            if lat[i] == 0:
                temp_height.append(height[i] - ground_level)
                temp_li.append(li[i])
                k += 1
                i += 1
            else:
                # combining height and weather mini lists for storage
                temp_combo_li = combineLatLon(temp_height, temp_li)

                # making sure first two heights aren't the same
                if temp_combo_li[0][0] == temp_combo_li[1][0]:
                    temp_combo_li.pop(0)

                # TEMPORARY TEST-forcing list to be a certain length
                # while len(temp_combo_li) > 20:
                    # temp_combo_li.pop()

                # getting to next latlon value in big list if not already there
                # while lat[i] == 0:
                    # i += 1
                    # k += 1

                # key is location of previous latlon in big list
                key = '%i, %i' % (lat[i-k], lon[i-k])

                # appending mini-list to dictionary at latlon key
                if d:
                    data[key][keyName] = temp_combo_li
                else:
                    data[key] = {keyName: temp_combo_li}

                # clearing mini-list and restarting
                temp_height = []
                temp_li = []
                temp_combo_li = []
                k = 0
                temp_height.append(height[i])
                ground_level = temp_height[0]
                ground_altitudes.append(ALT - ground_level)
                temp_height[0] = temp_height[0] - ground_level
                temp_li.append(li[i])
                k += 1
                i += 1
        # getting first element in big list
        else:
            temp_height.append(height[i])
            ground_level = temp_height[0]

            ground_altitudes = [ALT - ground_level]
            temp_height[0] = temp_height[0] - ground_level
            temp_li.append(li[i])
            k += 1
            i += 1

    # getting data from final mini-list
    temp_combo_li = combineLatLon(temp_height, temp_li)
    # making sure first two heights aren't the same
    if temp_combo_li[0][0] == temp_combo_li[1][0]:
        temp_combo_li.pop(0)

    # while len(temp_combo_li) > 20:
        # temp_combo_li.pop()

    # dictionary key
    key = '%i, %i' % (lat[i-k], lon[i-k])

    # making dictionary
    if d:
        data[key][keyName] = temp_combo_li
    else:
        data[key] = {keyName: temp_combo_li}

    return data, ground_altitudes


def combineLatLon(lat, lon):
    '''combineLatLon takes a list of latitudes and a list of longitudes
    that are the same length and combines them into a double list.
    '''
    w_latlon = []
    for i in range(len(lat)):
        w_latlon.append([lat[i], lon[i]])

    return w_latlon


def read_input(filename):
    # Read inputs from a file
    f = open(filename, 'r')
    line = f.read()
    line = line.split('\t')
    f.close()

    # Collect input values
    inputs = []
    for i in range(len(line)-1):
        inputs.append(float(line[i]))

    nBumps = inputs[0]  # this input will denote the number of bumps
    bump_inputs = []  # initialize
    if nBumps >= 1:
        for i in range(1, int(nBumps*3+1), 3):
            height = inputs[i]
            length_down_body = inputs[i+1]
            width = inputs[i+2]
            bump = [height, length_down_body, width]
            bump_inputs.append(bump)
    else:
        raise RuntimeError(
            "The first input (denoting the number of bumps) must be an integer greater than or equal to 1")
    return bump_inputs
