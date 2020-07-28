from rapidboom import AxieBump, EquivArea
from weather.boom import read_input
from weather.scraper.twister import process_data
# import tensorflow as tf
import platform
from weather.parameterize_atmosphere.autoencoder import *
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import csv

################################################################################
#                               Set Constants
################################################################################
cruise_alt = 50000 #ft
run_method = 'EquivArea'
bump_inputs = [0,0,0] # we don't want any bumps, just use baseline geometry
CASE_DIR = './'
n= 75 # number of interpolated points
model = '25D'

################################################################################
#                               Prepare results file
################################################################################
list = ['Profile','Loudness']
with open('./PLdB_predictions.csv', mode='a+') as file:
    file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    file_writer.writerow(list)

################################################################################
# 			       Prepare weather data for AxieBump and EquivArea
################################################################################
raw_data = pd.read_csv("./autoencoder_results_predictions.csv",skiprows=0)
raw_data = pd.DataFrame.to_numpy(raw_data)
elevations = raw_data[:,3]

temperature_profiles = raw_data[:,4:4+n]
humidity_profiles = raw_data[:,4+n:]

for profile in range(4196,len(elevations)):
    print("Profile: ",profile+1)
    elevation = elevations[profile] # grab the specific profile elevation
    altitudes = np.linspace(elevation, elevation+cruise_alt, n) # generate altitude vector

    # move this to out of big loop later
    main_t = []
    main_h = []
    # # amount of temperature or humidity points each
    # points = 75
    # change to len(raw_data)
    i=profile
    prof_t = []
    prof_h = []
    for x in range(n):
        both_t = []
        both_h = []
        both_t.append(altitudes[x])
        both_t.append(temperature_profiles[i][x])
        prof_t.append(both_t)
        both_h.append(altitudes[x])
        both_h.append(humidity_profiles[i][x])
        prof_h.append(both_h)
    main_t.append(prof_t)
    main_h.append(prof_h)
        
    ###
    weather_data = {"humidity":(main_h[0]),
                    "temperature":(main_t[0])}

    ################################################################################
    #                            Run sBoom
    ################################################################################

    print(platform.system())
    if platform.system() == 'Linux' or platform.system() == 'Darwin':
        PANAIR_EXE = 'panair'
        SBOOM_EXE = 'sboom_linux'
    elif platform.system() == 'Windows':
        PANAIR_EXE = 'panair.exe'
        SBOOM_EXE = 'sboom.exe'
    else:
        raise RuntimeError("platfrom not recognized")

    # Run
    if run_method == 'panair':
        axiebump = AxieBump(CASE_DIR, PANAIR_EXE, SBOOM_EXE,
                        altitude=cruise_alt,
                        weather=weather_data)
        axiebump.MESH_COARSEN_TOL = 0.00045
        axiebump.N_TANGENTIAL = 20
        loudness = axiebump.run(bump_inputs)
    elif run_method == 'EquivArea':
        if model=="25D": 
            axiebump = EquivArea(CASE_DIR, SBOOM_EXE,
                        altitude=cruise_alt,
                        elevation = elevation, 
                        weather=weather_data,
                        area_filename = '41N_74W_25D_adapt07_EALINENEW4.dat',
                        mach=1.6, phi=0)
        elif model=="X59":
            axiebump = EquivArea(CASE_DIR, SBOOM_EXE,
                        altitude=cruise_alt,
                        weather=weather_data,
                        area_filename = 'x_59_mach1p400_aoa0p000_phi00p00_clean.eqarea',
                        mach=1.4, phi=0)
            axiebump.REF_LENGTH = 27.432
            axiebump.R_over_L = 3
        else:
            raise RuntimeError("model not recognized")
        loudness = axiebump.run(bump_inputs)
    else:
        raise RuntimeError("evaluation method not recognized")

    print("Predicted perceived loudness", loudness)
    # print("Pre-calculated perceived loudness", PLdB[profile])
    
    list = [profile+1,loudness]
    with open('./PLdB_predictions.csv', mode='a+') as file:
        file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        file_writer.writerow(list)

# # write output file for Matlab to read
# f = open('axie_bump_outputs.txt', 'w')
# f.write('%6.5f\t' % (loudness))
# f.close()
