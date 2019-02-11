import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.stats import gaussian_kde
from weather import process_noise


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

# Get noise data
filename = "../data/noise/noise_per_county"
noise_data = pickle.load(open(filename + '.p', 'rb'))
noise = copy.deepcopy(noise_data[:, 3])
pop = noise_data[:, 2]/sum(noise_data[:, 2])
annoyance = exterior_annoyance(noise)

# Organizing data
hist, bins = np.histogram(noise, bins=np.arange(79, 89, .1), density=True)
inds = np.digitize(noise, bins)
pop_level = np.zeros(len(bins))
annoyance_level = np.zeros(len(bins))
print(len(noise), len(annoyance))
print(len(hist), len(bins))
for i in range(len(inds)):
    pop_level[inds[i]-1] += pop[i]
    annoyance_level[inds[i]-1] += pop[i]*annoyance[i]/100.
print('bins', bins)
print('pop', pop_level)
fig, ax1 = plt.subplots()

# ax2.plot(z, z2, 'r')
ax1.bar(bins, pop_level, color='b', width=0.4, label='Population')
ax1.bar(bins, annoyance_level, color='r', width=0.4, label='Population annoyed')
ax1.set_ylabel('Perceived loudness/Annoyance distribution', color='r')
ax1.tick_params('y', colors='r')

kde = gaussian_kde(noise)
x = np.linspace(79, 89, 200)
ax1.plot(x, kde(x), '--k', lw=3, label='PL probability')
ax1.set_xlabel('PL')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Probability', color='k')
ax1.tick_params('y', colors='k')

fig.tight_layout()
plt.legend()
plt.show()
