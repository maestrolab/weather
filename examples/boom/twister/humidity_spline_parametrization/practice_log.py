import numpy as np
import matplotlib.pyplot as plt
from misc_humidity import prepare_standard_profiles, calculate_vapor_pressures, package_data

path = './../../../../data/weather/standard_profiles/standard_profiles.p'
standard_profiles = prepare_standard_profiles(standard_profiles_path=path)

rh_alts, rh = package_data(standard_profiles['relative humidity'])
temp_alts, temp = package_data(standard_profiles['temperature'])
pres_alts, pres = package_data(standard_profiles['pressure'])

a_vps, sat_vps = calculate_vapor_pressures(rh, temp, pres)

plt.plot(np.log(rh_alts), a_vps)
# plt.plot(rh_alts, a_vps)
plt.xlabel('log(Altitude)')
plt.show()

#
# x = np.array([1,2,3,4,5])
# y = -x+5
#
# x_e = np.exp(x)
#
# fig = plt.figure()
# plt.plot(x,y)
#
# fig = plt.figure()
# plt.plot(x_e,y)
#
# plt.show()
