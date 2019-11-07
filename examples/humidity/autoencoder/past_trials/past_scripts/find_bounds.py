import pickle
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from autoencoder import *

# Load profile data
balloon_data  ='../../balloon_data/2017+2018/US_2017_2018'
# balloon_data  ='balloon_data/72469_all_years/72469_2017'
data = pickle.load(open(balloon_data + '.p','rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])

# Interpolate profiles
n = 75
alt_interpolated = np.linspace(0,13500,n)
rh, temp = interpolate_profiles(alt_interpolated, rh, temp)

rh_maxes = np.array([np.max(r_list) for r_list in rh[:,:,1]])
rh_mins = np.array([np.min(r_list) for r_list in rh[:,:,1]])
temp_maxes = np.array([np.max(t_list) for t_list in temp[:,:,1]])
temp_mins = np.array([np.min(t_list) for t_list in temp[:,:,1]])

print('RH Max:(mean, standard deviation, lower bound and then upper bound)')
print(np.mean(rh_maxes))
print(np.std(rh_maxes))
print(np.mean(rh_maxes) - 3*np.std(rh_maxes))
print(np.mean(rh_maxes) + 3*np.std(rh_maxes))
print('RH Min:')
print(np.mean(rh_mins))
print(np.std(rh_mins))
print(np.mean(rh_mins) - 3*np.std(rh_mins))
print(np.mean(rh_mins) + 3*np.std(rh_mins))

print('Temp Max:(mean, standard deviation, lower bound and then upper bound)')
print(np.mean(temp_maxes))
print(np.std(temp_maxes))
print(np.mean(temp_maxes) - 3*np.std(temp_maxes))
print(np.mean(temp_maxes) + 3*np.std(temp_maxes))
print('Temp Min:')
print(np.mean(temp_mins))
print(np.std(temp_mins))
print(np.mean(temp_mins) - 3*np.std(temp_mins))
print(np.mean(temp_mins) + 3*np.std(temp_mins))
asdf

rh = np.concatenate(rh[:,:,1])
temp = np.concatenate(temp[:,:,1])

# Initialize dataframe
df = pd.DataFrame({'humidity':rh,
                   'temperature':temp})


# Compute statistical metrics
rh_mean = df.humidity.mean()
temp_mean = df.temperature.mean()
rh_std = df.humidity.std()
temp_std = df.temperature.std()

rh_1 = rh_mean + rh_std
temp_1 = temp_mean + temp_std
rh_2 = rh_mean + rh_std*2
temp_2 = temp_mean + temp_std*2

rh_1_ = rh_mean - rh_std
temp_1_ = temp_mean - temp_std
rh_2_ = rh_mean - rh_std*2
temp_2_ = temp_mean - temp_std*2

# Find means of upper and lower 50% of the data (then find std and +/-)
rh_cutoff = rh_mean-rh_std*0.5
rh_lower = np.array([r for r in df.humidity if r < rh_cutoff])
rh_upper = np.array([r for r in df.humidity if r > rh_cutoff])
temp_lower = np.array([t for t in df.temperature if t < temp_mean])
temp_upper = np.array([t for t in df.temperature if t > temp_mean])

print(np.mean(rh_lower)-np.std(rh_lower))
print(np.mean(rh_upper)+np.std(rh_upper))
print(np.mean(temp_lower)-2*np.std(temp_lower))
print(np.mean(temp_upper)+2*np.std(temp_upper))

# Plot normal bell curve
# mu = rh_mean
# sigma = rh_std
# x = np.linspace(mu-3*sigma, mu+3*sigma, 1000)
# plt.plot(x, stats.norm.pdf(x,mu,sigma))
# plt.show()

# Plot box plot
fig, ax1 = plt.subplots()
plt.hist(rh, bins = 20, color = 'dodgerblue', edgecolor = 'k')
# plt.xlabel('Relative Humidity [%]')
# plt.ylabel('Frequency')
#
# plt.figure()
# plt.boxplot(rh, labels = ['Relative Humidity'])
# plt.show()

# PLot normal bell curves
# ax2 = ax1.twinx()
# mu = rh_mean
# sigma = rh_std
# mu_l = np.mean(rh_lower)
# sigma_l = np.std(rh_lower)
# mu_u = np.mean(rh_upper)
# sigma_u = np.std(rh_upper)
# mu = temp_mean
# sigma = temp_std
# mu_l = np.mean(temp_lower)
# sigma_l = np.std(temp_lower)
# mu_u = np.mean(temp_upper)
# sigma_u = np.std(temp_upper)
# x = np.linspace(mu-3*sigma, mu+3*sigma, 1000)
# plt.plot(x, stats.norm.pdf(x, mu_l, sigma_l), color = 'k')
# plt.plot(x, stats.norm.pdf(x, mu_u, sigma_u), color = 'k')
# plt.plot(x, stats.norm.pdf(x, mu_l, sigma_l)+stats.norm.pdf(x, mu_u, sigma_u))
# # plt.xlim(0,100)
# fig.tight_layout()
plt.show()
