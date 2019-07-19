import numpy as np
import matplotlib.pyplot as plt

path = 'noise_txt_files/temp_append_encoding_dim_varied/'
# n_params = [7, 8]
# path = 'noise_txt_files/temp_append_encoding_dim_varied/different_locations/'
# path = 'noise_txt_files/multiple_locations/'
n_params = [7, 9]
dim = 2500

# Compute the mean difference in PLdB and the max and min PLdB for each model
pldb = {n:{} for n in n_params}
average = np.zeros(len(n_params))
std = np.zeros(len(n_params))
for i in range(len(n_params)):
    if i == 0:
        filename = '%i_params_%i_dim_temp_append_constant_mins' % (n_params[i], dim)
    else:
        filename = '../7_12_19_non_trained_data'
    # filename = '72469_0-150'
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
asdf
# Plot all points per number of parameters used to describe the profiles
plt.figure()
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
        plt.scatter(n_params[i], n, color = 'dodgerblue')
plt.xlabel('Number of Parameters')
plt.ylabel('$\Delta$PLdB')
plt.xlim(6, 10)

# Plot lines at 0.5 and -0.5 for visual
line_0_x = [6+i for i in range(5)]
line_0_y = [0.5 for j in range(len(line_0_x))]
line_1_y = [-0.5 for j in range(len(line_0_x))]
plt.plot(line_0_x, line_0_y, color = 'k', linestyle = '--')
plt.plot(line_0_x, line_1_y, color = 'k', linestyle = '--')

# Set x tick labels
locs = n_params
labels = [int(n) for n in n_params]
plt.xticks(locs, labels)

for i in range(len(n_params)):
    print(list(outside_bounds.keys())[i])
    print(outside_bounds[n_params[i]])

plt.show()
