import pickle
import numpy as np
from weather.boom import boom_runner

def interpolate_temperature(temperature_altitudes, temperature_profile, reference_altitudes, reference_profile):
    '''interpolate_temperature creates a temperature profile that is made up
    of [atltidue, temperature] pairs at the altitudes contained within the
    reference_profile'''
    def find_indeces(point, data_set):
        '''find_indeces finds the bounding indices of data_set that include
        the point'''
        for i in range(len(data_set)):
            if point > data_set[i]:
                i1 = i
                i2 = i+1

        return i1, i2

    interpolated_temperature_profile = [temperature_profile[1]]
    for ref_alt in reference_altitudes[1:]:
        i1, i2 = find_indeces(ref_alt, temperature_altitudes)
        x = temperature_altitudes[i1:i2+1]
        y = temperature_profile[i1:i2+1]
        x0 = ref_alt
        interpolated_temperature = np.interp(x0,x,y)
        interpolated_temperature_profile.append(interpolated_temperature)

    return reference_altitudes, interpolated_temperature_profile

def package_data(data1, data2=None, method='unpack'):
    '''package_data packs or unpacks data in the form [[data1, data2]]'''
    if method == 'pack':
        packed_data = [[data1[i],data2[i]] for i in range(len(data1))]
        return packed_data
    elif method == 'unpack':
        unpacked_data_1 = [d[0] for d in data1]
        unpacked_data_2 = [d[1] for d in data1]
        return unpacked_data_1, unpacked_data_2

def calculate_vapor_pressures(temperature_profile, humidity_profile):
   '''calculate_vapor_pressures calculates the saturation_vapor_pressures
   and the actual_vapor_pressures given a temperature and humidity
   profile'''
   saturation_vapor_pressures = [0.61121*np.exp((18.678-(t/234.5))*
                                 (t/(257.14+t))) for t in
                                 temperature_profile]
   actual_vapor_pressures = [humidity_profile[i]/100*
                             saturation_vapor_pressures[i] for i in
                             range(len(humidity_profile))]

   return saturation_vapor_pressures, actual_vapor_pressures

def filter_vapor_pressures(saturation_vapor_pressures, actual_vapor_pressures):
    '''filter_vapor_pressures makes sure that the actual_vapor_pressures
    do not fall out of range of the saturation_vapor_pressures'''
    for i in range(len(actual_vapor_pressures)):
        if actual_vapor_pressures[i] > saturation_vapor_pressures[i]:
            actual_vapor_pressures[i] == saturation_vapor_pressures[i]
        elif actual_vapor_pressures[i] < 0:
            actual_vapor_pressures[i] == 0

    return actual_vapor_pressures

def calculate_humidity_profile(vapor_pressures, saturation_vapor_pressures):
    '''calculate_humidity_profile calculates the relative humidity values
    for all vapor pressure values in vapor_pressures'''
    relative_humidity_profile = [100*vapor_pressures[i]/
                                 saturation_vapor_pressures[i] for i in
                                 range(len(vapor_pressures))]

    return relative_humidity_profile

class deformation:
    '''Create a deformation to add to humidity profiles.

        Inputs:
         - base: base of deformation [m]
         - height: magnitude of desired deformation [kPa]
         - peak: altitude value at which to add deformation [m]
    '''

    def __init__(self, base, height, peak):
        self.base = base
        self.height = height
        self.peak = peak

    def _trapezoidal_area(self, x, y):
        """Compute the Riemann Sum using trapezoids."""
        areas = [(y[i-1]+y[i])*(x[i]-x[i-1])/2 for i in range(1, len(x))]
        total_area = sum(areas)

        return total_area

    def add_spike(self, altitudes, humidity_profile, temperature_profile):
        '''add_spike adds a spike to an inputted humidity profile without
        changing the integral of the humidity profile'''
        saturation_vps, spiked_vapor_profile = calculate_vapor_pressures(temperature_profile,
                                                               humidity_profile)
        for i in range(len(altitudes)):
             if altitudes[i] > self.peak-(self.base/2) and\
                altitudes[i] <= self.peak:
                 spiked_vapor_profile[i] = (2*self.height/self.base)*(altitudes[i]-
                                          (self.peak-(self.base/2)))
             elif altitudes[i] < self.peak+(self.base/2) and\
                  altitudes[i] > self.peak:
                 spiked_vapor_profile[i] = (2*self.height/self.base)*\
                                       (self.peak-altitudes[i]+self.base/2)

        spiked_vapor_profile = filter_vapor_pressures(saturation_vps,
                                                      spiked_vapor_profile)

        spiked_humidity_profile = calculate_humidity_profile(spiked_vapor_profile,
                                                             saturation_vps)

        return spiked_humidity_profile

    def change_average(self, altitudes, humidity_profile, temperature_profile, vapor_multiplier, original_humidity_profile=None):
        '''change_average multiplies the inputted humidity_profile by a constant
        multiplier'''

        saturation_vps, actual_vps = calculate_vapor_pressures(temperature_profile,
                                                               humidity_profile)
        area_of_profile = self._trapezoidal_area(altitudes, actual_vps)

        # Correction factor for area added by spike deformation
        if original_humidity_profile:
            original_saturation_vps, original_actual_vps = \
                            calculate_vapor_pressures(temperature_profile,
                                                      original_humidity_profile)
            original_area = _trapezoidal_area(altitudes, original_actual_vps)
        else:
            original_area = area_of_profile
        correction_factor = original_area/area_of_profile

        # Create scaled humidity profile
        scaled_vapor_profile = [actual_vps[i]*vapor_multiplier*correction_factor
                                for i in range(len(actual_vps))]
        scaled_vapor_profile = filter_vapor_pressures(saturation_vps,
                                                      scaled_vapor_profile)

        # Convert scaled vapor pressure profile to a scaled relative humidity profile
        scaled_humidity_profile = calculate_humidity_profile(scaled_vapor_profile,
                                                             saturation_vps)

        return scaled_humidity_profile

################################################################################
from optimization_tools.DOE import DOE
import time
import pickle

# Load data and organize sBoom_data
path = './../../../data/weather/twister/standard_profiles/standard_profiles.p'
standard_profiles = pickle.load(open(path,'rb'))

# Interpolate temperature values at altitudes in relative humidity profile
temperature_altitudes, temperatures = package_data(standard_profiles['temperature'])
rh_altitudes, rhs = package_data(standard_profiles['relative humidity'])

temperature_altitudes, temperatures = interpolate_temperature(
                                        temperature_altitudes, temperatures,
                                        rh_altitudes, rhs)

standard_profiles['temperature'] = package_data(temperature_altitudes,
                                                temperatures, method='pack')

cruise_altitude = 13000/0.3048 # [ft]
sBoom_data = {'profiles':[standard_profiles['temperature'],0,
              standard_profiles['relative humidity']],
              'height_to_ground':cruise_altitude}

def humidity_profile_DOE(inputs):
    deform = deformation(base=inputs['base'],height=inputs['height'],
                         peak=inputs['peak'])
    altitudes, humidity = package_data(inputs['sBoom']['profiles'][2],
                                         method='unpack')
    altitudes, temperature = package_data(inputs['sBoom']['profiles'][0],
                                         method='unpack')
    spiked_humidity = deform.add_spike(altitudes, humidity, temperature)
    scaled_humidity_profile = deform.change_average(altitudes, spiked_humidity,
                                         temperature, inputs['amount_of_vapor'])

    final_profile = package_data(altitudes, scaled_humidity_profile, method='pack')
    inputs['sBoom']['profiles'] = inputs['sBoom']['profiles'][:2] + [final_profile]

    noise = boom_runner(inputs['sBoom']['profiles'],
                        inputs['sBoom']['height_to_ground'])

    return {'noise':noise}

# Define points
problem = DOE(levels=5, driver='Full Factorial')
problem.add_variable('amount_of_vapor', lower=0.75, upper=1.25, type=float) # [%]
problem.add_variable('base', lower=0, upper=5000, type=float) # [m]
problem.add_variable('height', lower=0, upper=15, type=float) # [kPa]
problem.add_variable('peak', lower=0, upper=10000, type=float) # [m]
problem.define_points()

# Run for a function with dictionary as inputs
problem.run(humidity_profile_DOE, cte_input={'sBoom': sBoom_data})

problem.find_influences(not_zero=True)
problem.find_nadir_utopic(not_zero=True)
print('Nadir: ', problem.nadir)
print('Utopic: ', problem.utopic)

# Plot factor effects
problem.plot(xlabel=['amount_of_vapor', 'base', 'height', 'peak'],
             ylabel=['PLdB'], number_y=5)
