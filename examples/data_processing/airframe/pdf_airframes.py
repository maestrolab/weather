import pickle
import scipy.io
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.integrate import simps

from weather.scraper.flight_conditions import properties, Airframe

# Define object
# fuel = 56*6.01*0.4535
# initial_mass = 1111
# final_mass = initial_mass-fuel
# C172_props = properties({'Cl_alpha': 5.143, 'Cl_0': 0.31,
                         # 'planform': 16.1651, 'density': 0.770488088,
                         # 'mass_min': final_mass, 'mass_max': initial_mass,
                         # 'incidence': 0.})
# C172 = Airframe(airframe='C172', timestamp=1549036800,
                # filepath='../../data/flight_plan/v_aoa_pickles/icao24s_',
                # properties=C172_props)
# C172.retrieve_data(load_data=True)
# C172.train_pdf(1000)
C172 = pickle.load(open('C172.p', 'rb'))
C172.retrieve_data()
training = True
counter = 1
while training:
    try:
        print('Try %i' % counter)
        C172.train_pdf(4000)
        training = False
    except:
        counter += 1
        training = True
        
aoa, velocity = C172.normalize(np.array([[0, 20], [12, 65]])).T

C172.plot_pdf(xgrid = np.linspace(aoa[0], aoa[1], 400),
              ygrid = np.linspace(velocity[0], velocity[1], 400))
# Plot histograms

def constraint(sample):
    aoa, velocity = C172.denormalize(sample).T
    
    weight = np.ones(1000)*C172.properties.weight_max
    aoa_max = C172.calculate_aoa(weight, velocity)
    
    weight = np.ones(1000)*C172.properties.weight_min
    aoa_min = C172.calculate_aoa(weight, velocity)

    if aoa> aoa_min and aoa < aoa_max and aoa> 0 and aoa < 12 and velocity > 20 and velocity < 65:
        return False
    else:
        return True 
parameters = []
for i in range(150):
    sample = C172.pdf.sample(1)
    while constraint(sample):
        sample = C172.pdf.sample(1)
    parameters.append(sample)
C172.samples = np.array(parameters)
pickle.dump(C172, open("c172_new.p", "wb"))

weight = np.ones(1000)*C172.properties.weight_max
velocity = np.linspace(0, 1, 1000)
_aoa, velocity = C172.denormalize(np.array([velocity, velocity]).T).T
aoa = C172.calculate_aoa(weight, velocity)
# aoa, velocity = C172.normalize(np.array([aoa, velocity]).T).T
plt.plot(aoa, velocity, '--w', label='Maximum weight')

weight = np.ones(1000)*C172.properties.weight_min
velocity = np.linspace(0, 1, 1000)
_aoa, velocity = C172.denormalize(np.array([velocity, velocity]).T).T
aoa = C172.calculate_aoa(weight, velocity)
# aoa, velocity = C172.normalize(np.array([aoa, velocity]).T).T
plt.plot(aoa, velocity, '--w', label='Minimum weight')

plt.xlabel(r'Angle of Attack ($^{\circ}$)')
plt.ylabel('Velocity (m/s)')
plt.legend()

x, y = C172.samples.T
aoa, velocity = C172.denormalize(C172.samples).T
plt.scatter(aoa, velocity, c='k')

x, y = C172.database.T
# aoa, velocity = C172.denormalize(C172.samples).T
plt.scatter(x, y, c='r')

plt.show()
BRAKE
# Calculating total probability
xgrid = np.linspace(-5, 35, 1000)
ygrid = np.linspace(20, 75, 1000)
X, Y = np.meshgrid(xgrid, ygrid)
Z = np.exp(C172.pdf.score_samples(C172.normalize(np.array([X.ravel(), Y.ravel()]).T)))
Z = np.reshape(Z, X.shape)
total_list = []
for i in range(len(Z)):
    # print('X', X[i])
    # print('Y', Y[:, 0])
    # print('Z', Z[i, :])
    numerator = simps(Z[i, :], X[i])
    total_list.append(numerator)
total = simps(total_list, Y[:, 0])
print('Probability total', total)

# # Max/Min/Average plot
# weight = np.ones(1000)*C172.properties.weight_max
# velocity = np.linspace(20, 75, 1000)
# aoa = C172.calculate_aoa(weight, velocity)
# plt.plot(aoa, velocity, '--b', label='Maximum weight')

# weight = np.ones(1000)*C172.properties.weight_min
# velocity = np.linspace(20, 75, 1000)
# aoa = C172.calculate_aoa(weight, velocity)
# plt.plot(aoa, velocity, '--b', label='Minimum weight')

# weight = np.ones(1000)*.5*(C172.properties.weight_max+C172.properties.weight_min)
# velocity = np.linspace(20, 75, 1000)
# aoa = C172.calculate_aoa(weight, velocity)
# plt.plot(aoa, velocity, 'b', lw=2, label='Average')
# plt.xlabel(r'Angle of Attack ($^{\circ}$)')
# plt.ylabel('Velocity (m/s)')
# plt.legend()
# plt.show()

# # Plot histograms
# log_dens = C172.pdf_velocity.score_samples(velocity.reshape(len(velocity), 1))
# samples = C172.pdf_velocity.sample(len(C172.velocity))
# plt.figure(1)
# plt.hist(samples, bins=20, label='Sampled')
# plt.hist(C172.velocity, bins=20, label='Real')
# plt.legend()
# plt.figure(2)
# plt.plot(velocity, np.exp(log_dens))
# plt.xlabel('Velocity(m/s)')
# plt.ylabel('Probability density function')
# plt.show()

# Plot PDF
C172.plot_pdf()

C172.samples = C172.pdf.sample(150)

weight = np.ones(1000)*C172.properties.weight_max
velocity = np.linspace(20, 100, 1000)
aoa = C172.calculate_aoa(weight, velocity)
plt.plot(aoa, velocity, '--w', label='Maximum weight')

weight = np.ones(1000)*C172.properties.weight_min
velocity = np.linspace(20, 100, 1000)
aoa = C172.calculate_aoa(weight, velocity)
plt.plot(aoa, velocity, '--w', label='Minimum weight')

plt.xlabel(r'Angle of Attack ($^{\circ}$)')
plt.ylabel('Velocity (m/s)')
plt.legend()

x, y = C172.samples.T
plt.scatter(x, y, c='k')

# x, y = C172.database.T
# plt.scatter(x, y, c='b')

plt.show()