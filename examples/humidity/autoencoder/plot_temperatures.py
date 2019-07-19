import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load profile data
data = pickle.load(open('./balloon_data/72469_2000-2018.p','rb'))
temp = np.array(data['temperature'])

n = 75
alt_interpolated = np.linspace(0,13500,n)
temps_to_average = np.zeros((len(temp), len(alt_interpolated)))

# plt.figure()
for i in range(len(temp)):
    alts, temps = np.array(temp[i]).T
    fun = interp1d(alts, temps)
    temp_interp = fun(alt_interpolated)
    # plt.plot(temp_interp, alt_interpolated, alpha = 0.1, color = 'k')

    temps_to_average[i] = temp_interp

average_profile = np.average(temps_to_average, axis = 0)
# plt.plot(average_profile, alt_interpolated, color = 'dodgerblue')
# plt.ylim(0,15000)
# plt.show()

print(average_profile[0])
print(average_profile[-1])
