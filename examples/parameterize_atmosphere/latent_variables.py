import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

################################################################################
#                                Parameters
################################################################################
n_params = 5

file_path = '../../data/atmosphere_models/noise/%i_parameters.txt' % (n_params)

if n_params == 4:
    n_range = 4
else:
    n_range = n_params-1
################################################################################
#                      Read and store latent variables
################################################################################
f = open(file_path, 'r')
latent_vars = {i:[] for i in range(n_range)}
lines = f.readlines()
for line in lines:
    for j in range(n_range):
        latent_vars[j].append(line.split()[j+4])

f.close()

################################################################################
#                  Convert the latent variables to floats
################################################################################
keys = latent_vars.keys()
for key in keys:
    latent_vars[key] = np.array(latent_vars[key]).astype(float)

################################################################################
#                          Plot latent variables
################################################################################
colors = ['dodgerblue', 'firebrick', 'aqua', 'k']
for i in range(n_range-1):
    x = [i for j in range(len(latent_vars[i]))]
    plt.scatter(x, latent_vars[i], color = colors[i])

loc = [int(i) for i in range(n_range)]
labels = loc
plt.xticks(loc, labels)
plt.xlabel('Latent Variable')
plt.ylabel('Value')

################################################################################
#                      Plot averages of latent variables
################################################################################
for i in range(n_range):
    plt.scatter(i, np.mean(latent_vars[i]), color = 'k')
plt.plot(loc, [np.mean(latent_vars[key]) for key in latent_vars.keys()],
         '--', color = 'k')

bp_data = np.array([np.array(latent_vars[key]) for key in latent_vars.keys()])

for i in range(n_range):
    min_val = np.min(bp_data[i])
    max_val = np.max(bp_data[i])
    print_statement = 'Latent variable %i: [%.3f, %.3f]' % (i, min_val, max_val)
    print(print_statement)

################################################################################
#                              Plot bsoxplots
################################################################################
print(bp_data.T)
plt.figure()
bp = plt.boxplot(bp_data.T, labels = list(map(str, range(n_range))))
plt.xlabel('Latent Variable')
plt.ylabel('Value')

plt.show()

################################################################################
#            Determine bounds (Q3 + 1.5*IQR) for each latent variable
################################################################################
for i in range(0,len(bp['caps'])-1,2):
    lb = bp['caps'][i].get_ydata()[0]
    ub = bp['caps'][i+1].get_ydata()[0]
    feature_bounds = np.array([lb, ub])
    print('Latent variable %i: %.3f to %.3f' % (int(i/2), lb, ub))
