import copy
import pickle
import numpy as np
from scipy import interpolate

try:
    from pyldb import perceivedloudness
    from rapidboom.sboomwrapper import SboomWrapper
except:
    print('Boom analysis is disabled')


def boom_runner(data, altitude_feet,
                nearfield_file='../../../data/nearfield/25D_M16_RL5.p'):
    '''
    Runs sBOOM
     Python3 Version
    '''

    # Define parameters
    CASE_DIR = "."  # folder where all case files for the tools will be stored
    REF_LENGTH = 32.92
    MACH = 1.6
    R_over_L = 5

    # weather data
    [temperature, wind, humidity] = data

    # get pressure signature from pickle
    nearfield_sig = pickle.load(open(nearfield_file, "rb"))

    # initialize sBOOM
    sboom = SboomWrapper(CASE_DIR, exe="sboom_windows.dat.allow")

    # change mach_number for each iteration based on wind
    mach = MACH  # MachModifier(DIRECTION, MACH, ALT, wind)

    # update sBOOM settings and run
    sboom.set(mach_number=mach,
              altitude=altitude_feet,
              propagation_start=R_over_L*REF_LENGTH*3.28084,
              altitude_stop=0.,
              output_format=0,
              input_xdim=2,
              signature=nearfield_sig,
              input_temp=temperature,
              input_wind=wind,
              input_humidity=humidity,
              propagation_points=40000,
              padding_points=8000)

    sboom_results = sboom.run()

    ground_sig = sboom_results["signal_0"]["ground_sig"]

    # grab the loudness level
    noise_level = perceivedloudness(ground_sig[:, 0], ground_sig[:, 1], pad_rear=4)

    return noise_level


def prepare_weather_sBoom(data, j):
    # Define latitude and longitude
    key = list(data.keys())[j]

    # temperature input (altitude ft, temperature F)
    temperature = data[key]['temperature']

    # wind input (altitude ft, wind X, wind Y)
    wind = []
    wind = data[key]['wind_x']  # data[key]['wind_y']]
    for i in range(len(wind)):
        wind[i].append(data[key]['wind_y'][i][1])

    # wind input (altitude ft, humidity %)
    humidity = data[key]['humidity']
    return(temperature, wind, humidity)


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
