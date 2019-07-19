import numpy as np
import matplotlib.pyplot as plt

path = 'noise_txt_files/encoding_dim_and_layer_dim_varied/'
n_params = [4, 5, 6, 7, 8, 9, 10]
dim = 500

# Compute the mean difference in PLdB and the max and min PLdB for each model
pldb = {n:{} for n in n_params}
ave_error = np.zeros(len(n_params))
average = np.zeros(len(n_params))
std = np.zeros(len(n_params))
for i in range(len(n_params)):
    filename = '%i_params_%i_dim' % (n_params[i], dim)
    f = open(path + filename + '.txt', 'r')
    lines = f.readlines()
    pldb[n_params[i]]['profile'] = np.zeros(len(lines))
    for j in range(len(lines)):
        pldb[n_params[i]]['profile'][j] = lines[j].split()[3]
    f.close()
    average[i] = np.average(np.abs(pldb[n_params[i]]['profile']))
    std[i] = np.std(np.abs(pldb[n_params[i]]['profile']))
    ave_error[i] = np.mean(np.abs(pldb[n_params[i]]['profile']))

# Plot averages with standard deviations
fig, ax = plt.subplots(1, 1)
eb = ax.errorbar(n_params, average, yerr = std, marker='', color='k', capsize=5,
                 elinewidth=2, markeredgewidth=2, ecolor='k',  ls='--')
plt.scatter(n_params, average, c='k')
plt.xlabel('Number of Parameters')
plt.ylabel('$\Delta$PLdB')
#
# # Plot something here
# plt.figure()
# for i in range(len(n_params)):
#     for n in pldb[n_params[i]]['profile']:
#         plt.scatter(n_params[i], n, color = 'dodgerblue')
# plt.xlabel('Number of Parameters')
# plt.ylabel('$\Delta$PLdB')

# Plot average error results
plt.figure()
plt.plot(n_params, ave_error, '-o', color = 'dodgerblue')
locs = n_params
labels = [int(n) for n in n_params]
plt.xticks(locs, labels)
plt.xlabel('Number of Parameters')
plt.ylabel('Average $\Delta$PLdB')
plt.show()

################################################################################
# JUST DON'T WANT TO ADD ANOTHER SCRIPT TO THE DIRECTORY
################################################################################
n_params = [7, 10]
dim = [500, 1000, 2500, 3750, 5000]
# Compute the mean difference in PLdB and the max and min PLdB for each model
pldb = {n:{} for n in dim}
ave_error = np.zeros(len(dim))
average = np.zeros(len(dim))
std = np.zeros(len(dim))
for i in range(len(dim)):
    filename = '%i_params_%i_dim' % (n_params[0], dim[i])
    f = open(path + filename + '.txt', 'r')
    lines = f.readlines()
    pldb[dim[i]]['profile'] = np.zeros(len(lines))
    for j in range(len(lines)):
        pldb[dim[i]]['profile'][j] = lines[j].split()[3]
    f.close()
    average[i] = np.average(np.abs(pldb[dim[i]]['profile']))
    std[i] = np.std(np.abs(pldb[dim[i]]['profile']))
    ave_error[i] = np.mean(np.abs(pldb[dim[i]]['profile']))

# Plot average error results
plt.figure()
plt.plot(dim, ave_error, '-o', color = 'dodgerblue')
locs = dim
labels = [int(n) for n in dim]
plt.xticks(locs, labels)
plt.xlabel('Number of Dimensions in First Layer')
plt.ylabel('Average $\Delta$PLdB')
plt.show()
