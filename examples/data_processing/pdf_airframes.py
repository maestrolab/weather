import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

from weather.scraper.flight_conditions import properties, Airframe

# Define object
C172_props = properties({'Cl_alpha': 5.143, 'Cl_0': 0.31,
                         'planform': 16.1651, 'density': 0.770488088,
                         'mass_min': 618., 'mass_max': 919.,
                         'incidence': 0.})
C172 = Airframe(airframe='C172', timestamp=1549036800,
                filepath='../../data/flight_plan/v_aoa_pickles/icao24s_',
                properties=C172_props)
C172.retrieve_data()
C172.train_pdf(1000)

# Calculating total probability
xgrid = np.linspace(-5, 35, 1000)
ygrid = np.linspace(20, 75, 1000)
X, Y = np.meshgrid(xgrid, ygrid)
Z = np.exp(C172.pdf.score_samples(np.array([X.ravel(), Y.ravel()]).T))
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

# Max/Min/Average plot
weight = np.ones(1000)*C172.properties.weight_max
velocity = np.linspace(20, 100, 1000)
aoa = C172.calculate_aoa(weight, velocity)
plt.plot(aoa, velocity, '--b', label='Maximum weight')

weight = np.ones(1000)*C172.properties.weight_min
velocity = np.linspace(20, 100, 1000)
aoa = C172.calculate_aoa(weight, velocity)
plt.plot(aoa, velocity, '--b', label='Minimum weight')

weight = np.ones(1000)*.5*(C172.properties.weight_max+C172.properties.weight_min)
velocity = np.linspace(20, 100, 1000)
aoa = C172.calculate_aoa(weight, velocity)
plt.plot(aoa, velocity, 'b', lw=2, label='Average')
plt.xlabel(r'Angle of Attack ($^{\circ}$)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.show()

# Plot histograms
log_dens = C172.pdf_velocity.score_samples(velocity.reshape(len(velocity), 1))
samples = C172.pdf_velocity.sample(len(C172.velocity))
plt.figure(1)
plt.hist(samples, bins=20, label='Sampled')
plt.hist(C172.velocity, bins=20, label='Real')
plt.legend()
plt.figure(2)
plt.plot(velocity, np.exp(log_dens))
plt.xlabel('Velocity(m/s)')
plt.ylabel('Probability density function')
plt.show()

# Plot PDF
C172.plot_pdf()
