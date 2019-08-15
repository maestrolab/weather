import numpy as np
import matplotlib.pyplot as plt

# path = 'locations_in_US/2018/noise/'
path = 'multi-year_vs_single_year/noise/'
n_params = [9]
dim = 2500

# locations = [['72456','72214','72403','72265','72582']]
# locations.append(['72562','72214','72645','72582','72672'])
locations = [['72562','72214','72645','72582','72672']]
locations.append(['72786','72558','72261','72363','74646'])

# Compute the mean difference in PLdB and the max and min PLdB for each model
pldb = {n:{} for n in n_params}
average = np.zeros(len(n_params))
std = np.zeros(len(n_params))
pldb[n_params[0]]['profile'] = np.array([])#np.zeros(100)
for location in locations:
    for i in range(len(n_params)):
        filename = '%i_params_' % (n_params[0]) + '_'.join(location)
        f = open(path + filename + '.txt', 'r')
        lines = f.readlines()
        # pldb[n_params[i]]['profile'] = np.zeros(len(lines))
        for j in range(len(lines)):
            pldb[n_params[i]]['profile'] = np.append(pldb[n_params[i]]['profile'], lines[j].split()[3])
        f.close()
        pldb[n_params[i]]['profile'] = pldb[n_params[i]]['profile'].astype(float)
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

for i in range(len(n_params)):
    print(list(outside_bounds.keys())[i])
    print(outside_bounds[n_params[i]])
# print(len(pldb[n_params[0]]['profile']))
list_full = [n for n in pldb[n_params[0]]['profile']]

plt.figure()
plt.hist(list_full, bins = 20, color = 'dodgerblue', edgecolor = 'k')
plt.xlabel('$\Delta$PLdB')
plt.ylabel('Frequency')

plt.show()
