import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.stats import gaussian_kde
import seaborn as sns
from scipy.integrate import trapz
from scipy.ndimage.filters import gaussian_filter1d



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


day = '21'
months = ['06', '12']
year = '2018'
hours = ['00', '12']
altitude = '50000'
min_noise = 65
max_noise = 81
n_noise = 500

# Get noise data
directory = "../../../data/noise/county/"

colors = ['k', 'b', 'r', 'g']
counter = 0

cdf_day = {}
cdf_geo_day = {}
for month in months:
    cdf_day[month] = np.zeros(n_noise)
    cdf_geo_day[month] = np.zeros(n_noise)
    for hour in hours:
        filename = year + month + day + '_' + hour + '_' + altitude + '.p'
        noise_data = pickle.load(open(directory + filename, 'rb'))
        noise = copy.deepcopy(noise_data[:, 3])
        pop = noise_data[:, 2]
        print('min/max', min(noise), max(noise))
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

        # kwargs = {'cumulative': True}
        # sns.distplot(x, hist_kws=kwargs, kde_kws=kwargs)
        # Plotting loudness data
        # fig, ax1 = plt.subplots()
        # ax1.bar(.5*(bin_edges[:-1] + bin_edges[1:]), pop_level,
                # width=step, color='r', alpha=0.5, linewidth=1,
                # edgecolor='r')
        # ax1.set_xlabel('Perceived Level in dB')
        # ax1.set_ylabel('Exposed probability', color='r')
        # ax1.tick_params('y', colors='r')
        # plt.xlim(min_noise, max_noise)
        # plt.ylim(0, 0.25)

        # # Plotting noise data
        # ax2 = ax1.twinx()
        # ax2.hist(noise, bins=bins, color='b', alpha=0.5, density=True, linewidth=1,
                 # edgecolor='b')
        # ax2.set_ylabel('Geographic Probability', color='b')
        # ax2.tick_params('y', colors='b')
        # plt.ylim(0, 0.25)
        # fig.tight_layout()
        # plt.show()

        x = []
        cdf = []
        cdf_geo = []
        print(for_excel)
        print(for_excel)
        for i in range(len(for_excel)):
            x.append(for_excel[i][0])
            if i == 0:
                cdf.append(for_excel[i][1])
                cdf_geo.append(for_excel[i][2])
            else:
                print(i, for_excel[0:i+1,1])
                cdf.append(trapz(for_excel[0:i+1,1], x))
                cdf_geo.append(trapz(for_excel[0:i+1,2], x))
        cdfsmoothed = gaussian_filter1d(cdf, sigma=2)
        plt.plot(x,cdfsmoothed, colors[counter], label = month + '_' + hour)
        # plt.plot(x,cdf_geo, colors[counter], linestyle = '--', label = month + '_' + hour)
        counter += 1
        
        cdf_day[month] += cdf
        cdf_geo_day[month] += cdf_geo
plt.legend()
plt.show()

plt.figure()
for i in range(2):
    month = months[i]
    cdfsmoothed = gaussian_filter1d(cdf_day[month]/2, sigma=2)
    plt.plot(x, cdfsmoothed, colors[i], label=month)
    plt.plot(x, cdf_geo_day[month]/2., colors[i], linestyle = '--')
plt.legend()
plt.show()
