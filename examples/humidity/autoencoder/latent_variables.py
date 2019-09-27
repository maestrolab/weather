import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Specify number of parameters
n_params = 9

# Read and store latent variables
path = 'multi-year_vs_single_year/noise/'
filename = '%i_params_1_trained_data' % n_params

f = open(path + filename + '.txt', 'r')

latent_vars = {i:[] for i in range(n_params-5)}

lines = f.readlines()
for line in lines:
    for j in range(n_params-5):
        latent_vars[j].append(line.split()[j+4])

f.close()

# Convert the latent variables to floats
keys = latent_vars.keys()
for key in keys:
    latent_vars[key] = np.array(latent_vars[key]).astype(float)

# Plot latent variables
colors = ['dodgerblue', 'firebrick', 'aqua', 'k']
for i in range(n_params-5):
    x = [i for j in range(len(latent_vars[i]))]
    plt.scatter(x, latent_vars[i], color = colors[i])

loc = [int(i) for i in range(n_params-5)]
labels = loc
plt.xticks(loc, labels)
plt.xlabel('Latent Variable')
plt.ylabel('Value')

# Plot averages of latent variables
for i in range(n_params-5):
    plt.scatter(i, np.mean(latent_vars[i]), color = 'k')
plt.plot(loc, [np.mean(latent_vars[key]) for key in latent_vars.keys()],
         '--', color = 'k')

# latent_vars[2] = np.delete(latent_vars[2], np.where(latent_vars[2] > 7))

# plt.figure()
# x = np.linspace(-3,10,1000)
# mu = np.array([np.mean(latent_vars[key]) for key in latent_vars.keys()])
# sigma = np.array([np.std(latent_vars[key]) for key in latent_vars.keys()])
# for i in range(n_params-5):
#     plt.plot(x, stats.norm.pdf(x, mu[i], sigma[i]), color = colors[i])

bp_data = np.array([np.array(latent_vars[key]) for key in latent_vars.keys()])

plt.figure()
bp = plt.boxplot(bp_data.T, labels = ['0','1','2','3'])
plt.xlabel('Latent Variable')
plt.ylabel('Value')

# plt.show()

# Determine bounds (Q3 + 1.5*IQR) for each latent variable
for i in range(0,len(bp['caps'])-1,2):
    lb = bp['caps'][i].get_ydata()[0]
    ub = bp['caps'][i+1].get_ydata()[0]
    feature_bounds = np.array([lb, ub])
    print('Latent variable %i: %.3f to %.3f' % (int(i/2), lb, ub))
