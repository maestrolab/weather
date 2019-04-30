import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.stats import gaussian_kde


def exterior_annoyance(PL):
    PL = np.asarray(PL)
    annoyance = np.zeros(PL.shape)
    for i in range(len(PL)):
        PL_0 = 72.412
        slope = 5.7410605
        if PL[i] <= PL_0:
            annoyance[i] = 0
        elif PL[i] < 89.866:
            annoyance[i] = slope*(PL[i]-PL_0)
        else:
            annoyance[i] = 100
    return(annoyance)


day = '18'
month = '06'
year = '2018'
hour = '12'

min_noise = 77.75
max_noise = 88.25
n_noise = 21

# Get noise data
filename = "../../data/noise/noise_per_county"
noise_data = pickle.load(open(filename + '.p', 'rb'))
noise = copy.deepcopy(noise_data[:, 3])
pop = noise_data[:, 2]

# Processing data into bins
step = (max_noise - min_noise)/n_noise
bins = np.arange(min_noise, max_noise + step, step)
hist, bin_edges = np.histogram(noise, bins=bins, density=True)

# Processing population data in regards to noise
inds = np.digitize(noise, bin_edges)
pop_level = np.zeros(len(bin_edges)-1)
for i in range(len(inds)):
    pop_level[inds[i]-1] += pop[i]
pop_level = pop_level/sum(pop_level*step)

print('Integral of noise probability', sum(hist)*step)
print('Integral of population probability', sum(pop_level)*step)

for_excel = np.array([.5*(bin_edges[:-1] + bin_edges[1:]), pop_level, hist]).T
for i in range(len(for_excel)):
    print(for_excel[i][0], for_excel[i][1], for_excel[i][2],)

# # Plotting loudness data
# fig, ax1 = plt.subplots()
# ax1.bar(.5*(bin_edges[:-1] + bin_edges[1:]), pop_level,
#         width=step, color='r', alpha=0.5, linewidth=1,
#         edgecolor='r')
# ax1.set_xlabel('Perceived Level in dB')
# ax1.set_ylabel('Exposed probability', color='r')
# ax1.tick_params('y', colors='r')
# plt.xlim(min_noise, max_noise)
# plt.ylim(0, 0.25)
#
# # Plotting noise data
# ax2 = ax1.twinx()
# ax2.hist(noise, bins=bins, color='b', alpha=0.5, density=True, linewidth=1,
#          edgecolor='b')
# ax2.set_ylabel('Geographic Probability', color='b')
# ax2.tick_params('y', colors='b')
# plt.ylim(0, 0.25)
# fig.tight_layout()
# plt.show()
