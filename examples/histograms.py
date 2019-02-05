import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy


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

filename = "../data/noise/noise_per_county"
noise_data = pickle.load(open(filename + '.p', 'rb'))
z = copy.deepcopy(noise_data[:, 3])
pop = noise_data[:, 2]/sum(noise_data[:, 2])
np.histogram()

fig, ax1 = plt.subplots()
hist, bins = np.histogram(z, bins=np.arange(76, 90, 1), density=True)
ax1.hist(z, bins=20, color='#0504aa', alpha=0.7, rwidth=0.85, density=True)

ax1.set_xlabel('PL')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Probability', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
z2 = exterior_annoyance(z)
ax2.plot(z, z2, 'r')
ax2.set_ylabel('% More Annoyed', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
plt.show()
