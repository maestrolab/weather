import copy
import pickle
import numpy as np
from scipy import interpolate

from pyldb import perceivedloudness
from rapidboom.sboomwrapper import SboomWrapper
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
    sboom = SboomWrapper(CASE_DIR)#, exe="sboom_windows.dat.allow")

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

    ############################################################################
    # parametrize the humidity profile
    humidity = parametrize_humidity(humidity)
    ############################################################################

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


def parametrize_humidity(humidity):
    """parametrize_humidity takes a humidity profile represented as a list of
    data points (each data point has a corresponding altitude stored with it)
    and returns a parametrized humidity profile. The goal of the parametrized
    humidity profile is to simplify the input to the optimization algorithm
    while mainting an accurate percieved noise level (+/- 0.1 PLdB).
    """
    ############################################################################
    # Clumps of data points are clustered together and average humidity and
    #   average altitude computed and inputted into sboom.
    # humidity_vals = [h[1] for h in humidity]
    # temp_vals = [t[0] for t in humidity]
    #
    # humidity_vals_trial = [humidity_vals[0]]
    # humidity_vals_trial.append(sum(humidity_vals[1:3])/2)
    # humidity_vals_trial.append(sum(humidity_vals[3:5])/2)
    # humidity_vals_trial.append(sum(humidity_vals[5:8])/3)
    # humidity_vals_trial.append(sum(humidity_vals[8:13])/5)
    # humidity_vals_trial.append(humidity_vals[13])
    #
    # temp_vals_trial = [temp_vals[0]]
    # temp_vals_trial.append(sum(temp_vals[1:3])/2)
    # temp_vals_trial.append(sum(temp_vals[3:5])/2)
    # temp_vals_trial.append(sum(temp_vals[5:8])/3)
    # temp_vals_trial.append(sum(temp_vals[8:13])/5)
    # temp_vals_trial.append(temp_vals[13])
    # # print(humidity_vals_trial, temp_vals_trial)

    # humidity = []
    # for i in range(len(humidity_vals_trial)):
    #     humidity.append([temp_vals_trial[i],humidity_vals_trial[i]])
    ############################################################################
    # Point added via linear interpolation between two points in profile
    #   Did not change perceived loudness --> linear interpolation used in
    #   sboom?
    # new_humidity = []
    # new_humidity = [humidity[i] for i in range(13)]
    # new_humidity.append([12750.0,67.92])
    # new_humidity.append(humidity[13])
    ############################################################################
    # Points removed if altitude was above a certain threshold
    #   Works to an extent --> does not reduce number of data points enough
    # threshold = 4000 # [ft]
    # new_humidity = [h for h in humidity if h[0] < threshold]
    ############################################################################
    # Points added separately to four corners and center of profile to study
    #   the effect each one has on the perceived loudness.
    #   Essentially looks at if certain humidity values effect perceived
    #       loudness more or less at different altitudes.
    #
    # # humidity_altitude
    # low_high = [15000.0,5.0] # humidity[-1]
    # low_low = [150.0,5.0] # humidity[1]
    # high_high = [15000.0,65.0] # humidity[-1]
    # high_low = [150.0,65.0] # humidity[1]
    #
    # new_humidity = humidity[:1]
    # new_humidity.append(high_low)
    # [new_humidity.append(h) for h in humidity[1:]]
    ############################################################################
    new_humidity = [humidity[0]]
    new_humidity.append(humidity[5])
    new_humidity.append(humidity[10])
    new_humidity.append(humidity[14])
    #new_humidity.append(humidity[15])
    new_humidity.append(humidity[-1])

    humidity_vals = [h[1] for h in new_humidity]
    alt_vals = [alt[0] for alt in new_humidity]

    # Plot the humidity profile to help visualize differences between
    #   parameterization methods.
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(humidity_vals, alt_vals)
    plt.xlabel('Relative Humidity')
    plt.ylabel('Altitude [m]') # Check if altitude is in feet or meters
    plt.grid(True)

    filename = 'Test_1.png'
    plt.savefig('./../../data/weather/parametrized_humidity_profiles/' +
                filename)

    print(new_humidity)
    return new_humidity
