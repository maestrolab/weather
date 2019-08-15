import numpy as np
import matplotlib.pyplot as plt

# path = 'validation_data/noise/'
# path = 'multi-year_vs_single_year/noise/'
path = 'constant_min_max/noise/'
n_params = [5]
dim = 2500

locations = ['72562','72214','72645','72582','72672']

# Compute the mean difference in PLdB and the max and min PLdB for each model
pldb = {n:{} for n in n_params}
average = np.zeros(len(n_params))
std = np.zeros(len(n_params))

for i in range(len(n_params)):
    filename = '%i_params_' % (n_params[i]) + '_'.join(locations)
    # filename = '%i_params_1_trained_data' % n_params[0]
    # filename = '%i_params_elevation_bound_0_trained_data' % n_params[0]
    f = open(path + filename + '.txt', 'r')
    lines = f.readlines()
    pldb[n_params[i]]['profile'] = np.zeros(len(lines))
    for j in range(len(lines)):
        pldb[n_params[i]]['profile'][j] = lines[j].split()[3]
    f.close()
    average[i] = np.average(np.abs(pldb[n_params[i]]['profile']))
    std[i] = np.std(np.abs(pldb[n_params[i]]['profile']))
print('\nAverages:            %s' % average)
print('Standard Deviations: %s\n' % std)

# Plot all points per number of parameters used to describe the profiles
outside_bounds = {n:{0.5:0,0.75:0,1:0,'>':0} for n in n_params}
for i in range(len(n_params)):
    for n in pldb[n_params[i]]['profile']:
        # Count number of profiles that produced PLdB differences less than a value
        if np.abs(n) < 0.5:
            outside_bounds[n_params[i]][0.5] += 1
        elif np.abs(n) < 0.75:
            outside_bounds[n_params[i]][0.75] += 1
        elif np.abs(n) < 1:
            outside_bounds[n_params[i]][1] += 1
        else:
            outside_bounds[n_params[i]]['>'] += 1
#         plt.scatter(n_params[i], n, color = 'dodgerblue')
# plt.xlabel('Number of Parameters')
# plt.ylabel('$\Delta$PLdB')
# # plt.xlim(6, 10)
#
# # Plot lines at 0.5 and -0.5 for visual
# line_0_x = [8+i for i in range(3)]
# line_0_y = [0.5 for j in range(len(line_0_x))]
# line_1_y = [-0.5 for j in range(len(line_0_x))]
# plt.plot(line_0_x, line_0_y, color = 'k', linestyle = '--')
# plt.plot(line_0_x, line_1_y, color = 'k', linestyle = '--')
#
# # Set x tick labels
# locs = n_params
# labels = [int(n) for n in n_params]
# plt.xticks(locs, labels)

for i in range(len(n_params)):
    print(list(outside_bounds.keys())[i])
    print(outside_bounds[n_params[i]])

list_full = [n for n in pldb[n_params[0]]['profile']]

plt.figure()
plt.hist(list_full, bins = 20, color = 'dodgerblue', edgecolor = 'k')
plt.xlabel('$\Delta$PLdB')
plt.ylabel('Frequency')

plt.show()
