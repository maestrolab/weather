import pickle
import random
import matplotlib.pyplot as plt

from autoencoder import *

################################################################################
#                           Load profile data
################################################################################
balloon_data  ='balloon_data/2017+2018/US_2017_2018'

data = pickle.load(open(balloon_data + '.p','rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])
elevations = np.array([n[0] for n in np.array(data['height'])])

# Interpolate profiles
cruise_alt = 13500
n = 75
alt_interpolated = np.linspace(0,cruise_alt,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated, rh, temp)

################################################################################
#                           Load profiles to plot
################################################################################
profiles = pickle.load(open('profiles.p','rb'))['profiles']

################################################################################
#                           Plot profiles
################################################################################
THIS WILL PLOT AND SHOW ALL PLOTS THAT HAVE INDICES SAVED IN THE PICKLE FILE ABOVE
path = 'profiles/'
for i in range(len(profiles)):
    fig = plt.figure()
    plt.plot(temp_interpolated[profiles[i],:,1], alt_interpolated)
    plt.xlabel('Temperature (\N{DEGREE SIGN}F)')
    plt.ylabel('Altitude (m)')
    plt.xlim([-110,100])
    # plt.savefig(path + '%i_%i' % (i, profiles[i]))
    plt.close(fig)
    plt.show()
