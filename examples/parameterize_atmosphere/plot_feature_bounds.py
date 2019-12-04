import pickle
import numpy as np
import matplotlib.pyplot as plt

data = pickle.load(open('../../data/atmosphere_models/feature_bounds.p','rb'))

features = np.arange(0,75)

# 'rh' or 'temp'
key = 'rh'

lbs = np.array([data[key][feature][0] for feature in features])
ubs = np.array([data[key][feature][1] for feature in features])

plt.figure()
plt.plot(lbs, features, '-o', markersize = 1.5, linewidth = 0.5,
         color = 'dodgerblue', label = 'Lower bounds')
plt.plot(ubs, features, '-o', markersize = 1.5, linewidth = 0.5,
         color = 'maroon', label = 'Upper bounds')

# plt.xlabel('Temperature (\N{DEGREE SIGN}F)')
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Feature')

plt.ylim([0,74])
plt.legend()

plt.show()
