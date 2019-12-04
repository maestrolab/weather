from rapidboom import AxieBump, EquivArea
from weather.boom import read_input
from weather.scraper.twister import process_data
import platform

import pickle
import numpy as np
import tensorflow as tf
from weather.parameterize_atmosphere.autoencoder import *

# Input parameters
variable_file = "test_file"

cruise_alt = 13500 # meters
index = 0 # profile index from text file

################################################################################
# 				      Read variables from text file
################################################################################
# Read from text file [l1, l2, (if l3), (if l4), elevation]
f = open(variable_file + ".txt","r")
lines = f.readlines()
f.close()

latent_space = np.array([lines[0].split()]).astype("float")
for line in lines[1:]:
	variable_list = np.array([line.split()])
	variable_list = variable_list.astype("float")
	latent_space = np.vstack((latent_space, variable_list))

################################################################################
# 				Load model based off of input variable format
################################################################################
model_path = "../../../../data/atmosphere_models/" +\
             "trained_models/%i_parameters" % len(latent_space[0])
try:
	decoder = tf.keras.models.load_model(model_path + "_D.h5")
	n = 75 # number of data points in the profile (it is a model parameter)
except:
	raise RuntimeError("model not found")

################################################################################
# 			     	Insert zeros if required by the model
################################################################################
if len(latent_space[0]) != 5:
	if len(latent_space[0]) == 3:
		splice_index = 2
	elif len(latent_space[0]) == 4:
		splice_index = 1
	zeros_to_add = np.zeros((len(latent_space),1))
	intermediate = np.hstack((latent_space[:,:splice_index],zeros_to_add))
	latent_space = np.hstack((intermediate, latent_space[:,splice_index:]))

################################################################################
# 			     	Build profiles using the trained decoder
################################################################################
feature_bounds = "../../../../data/atmosphere_models/feature_bounds.p"
variable_bounds = pickle.load(open(feature_bounds,'rb'))
predictions = decoder.predict(latent_space[:,:-1])
predicted_normalized = np.array([normalize_variable_bounds(predictions[i][:-1], n,
								  variable_bounds = variable_bounds,
								  type = 'both', inverse = True) for i in
								  range(latent_space.shape[0])])

################################################################################
# 			       Prepare weather data for AxieBump and EquivArea
################################################################################
altitudes = np.linspace(0, cruise_alt, n)
humidity_profiles = predicted_normalized[:,:n]
temperature_profiles = predicted_normalized[:,n:]

humidity = np.array([np.array([altitudes, humidity_profiles[i]]).T for i in
					 range(len(latent_space))])
temperature = np.array([np.array([altitudes, temperature_profiles[i]]).T for i in
					 range(len(latent_space))])

weather_data = {"humidity":list(humidity[index]),
                "temperature":list(temperature[index])}

height_to_ground = cruise_alt/0.3048 - latent_space[index,-1]# feet

################################################################################
#                            Jon's Code
################################################################################
deformation, run_method, bump_inputs = read_input('../axie_bump_inputs.txt')

CASE_DIR = "./"  # axie bump case
#PANAIR_EXE = 'panair.exe'  # name of the panair executable
#SBOOM_EXE = 'sboom_windows.dat.allow'  # name of the sboom executable

print(platform.system())
if platform.system() == 'Linux' or platform.system() == 'Darwin':
	PANAIR_EXE = 'panair'
	SBOOM_EXE = 'sboom_linux'
elif platform.system() == 'Windows':
	PANAIR_EXE = 'panair.exe'
	SBOOM_EXE = 'sboom_windows.dat.allow'
else:
	raise RuntimeError("platfrom not recognized")

# Run
if run_method == 'panair':
	axiebump = AxieBump(CASE_DIR, PANAIR_EXE, SBOOM_EXE,
                    altitude=height_to_ground,
                    weather=weather_data,
                    deformation=deformation)
	axiebump.MESH_COARSEN_TOL = 0.00045
	axiebump.N_TANGENTIAL = 20
	loudness = axiebump.run(bump_inputs)
elif run_method == 'EquivArea':
	axiebump = EquivArea(CASE_DIR, SBOOM_EXE,
                    altitude=height_to_ground,
                    weather=weather_data,
                    deformation=deformation,
                    mach=1.6, phi=0)
	loudness = axiebump.run(bump_inputs)
else:
	raise RuntimeError("evaluation method not recognized")

print("Perceived loudness", loudness)

# write output file for Matlab to read
f = open('axie_bump_outputs.txt', 'w')
f.write('%6.5f\t' % (loudness))
f.close()

#                     area_filename = 'mach1p600_aoa0p000_phi00p00_powered.eqarea',
