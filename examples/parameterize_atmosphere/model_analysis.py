import numpy as np
import matplotlib.pyplot as plt

################################################################################
#                                 Parameters
################################################################################
# Specify model to analyze based on the number of parameters used
n_params = [5]

file_path = '../../data/atmosphere_models/noise/%i_parameters.txt' % n_params[0]

################################################################################
#                  Mean and standard deviation calculations
################################################################################
pldb = {n:{} for n in n_params}
average = np.zeros(len(n_params))
std = np.zeros(len(n_params))

for i in range(len(n_params)):
    f = open(file_path, 'r')
    lines = f.readlines()
    pldb[n_params[i]]['profile'] = np.zeros(len(lines))
    for j in range(len(lines)):
        pldb[n_params[i]]['profile'][j] = lines[j].split()[3]
    f.close()
    average[i] = np.average(np.abs(pldb[n_params[i]]['profile']))
    std[i] = np.std(np.abs(pldb[n_params[i]]['profile']))
print('\nAverages:            %s' % average)
print('Standard Deviations: %s\n' % std)

################################################################################
#              Count number of profiles inside PLdB thresholds
################################################################################
outside_bounds = {n:{0.5:0,0.75:0,1:0,'>':0} for n in n_params}
for i in range(len(n_params)):
    for n in pldb[n_params[i]]['profile']:
        if np.abs(n) < 0.5:
            outside_bounds[n_params[i]][0.5] += 1
        elif np.abs(n) < 0.75:
            outside_bounds[n_params[i]][0.75] += 1
        elif np.abs(n) < 1:
            outside_bounds[n_params[i]][1] += 1
        else:
            outside_bounds[n_params[i]]['>'] += 1

for i in range(len(n_params)):
    print(list(outside_bounds.keys())[i])
    print(outside_bounds[n_params[i]])

################################################################################
#                         Plot histogram of results
################################################################################
list_full = [n for n in pldb[n_params[0]]['profile']]

plt.figure()
plt.hist(list_full, bins = 20, color = 'dodgerblue', edgecolor = 'k')
plt.xlabel('$\Delta$PLdB')
plt.ylabel('Frequency')

plt.show()
