import pickle
import numpy as np
from weather.boom import boom_runner

def package_data(data1, data2=None, method='unpack'):
    '''package_data packs or unpacks data in the form [[data1, data2]]'''
    if method == 'pack':
        packed_data = [[data1[i],data2[i]] for i in range(len(data1))]
        return packed_data
    elif method == 'unpack':
        unpacked_data_1 = [d[0] for d in data1]
        unpacked_data_2 = [d[1] for d in data1]
        return unpacked_data_1, unpacked_data_2

# Probably should save standard_profiles in current directory instead of
#   preparing them everytime this script is run.
def prepare_standard_profiles(cruise_altitude=13000, standard_profiles_path=
'./../../../data/weather/twister/standard_profiles/standard_profiles.p'):
       '''prepare_standard_profiles loads the standard profile data and organizes
       sBoom_data to be inputted into the humidity_profile_DOE'''
       def interpolate_temperature(temperature_altitudes, temperature_profile,
                                   reference_altitudes, reference_profile):
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

       standard_profiles = pickle.load(open(standard_profiles_path,'rb'))

       # Interpolate temperature values at altitudes in relative humidity profile
       temperature_altitudes, temperatures = package_data(standard_profiles['temperature'])
       rh_altitudes, rhs = package_data(standard_profiles['relative humidity'])

       temperature_altitudes, temperatures = interpolate_temperature(
                                                temperature_altitudes, temperatures,
                                                rh_altitudes, rhs)

       standard_profiles['temperature'] = package_data(temperature_altitudes,
                                                        temperatures, method='pack')

       cruise_altitude = cruise_altitude/0.3048 # [ft]
       sBoom_data = {'profiles':[standard_profiles['temperature'],0,
                      standard_profiles['relative humidity']],
                      'height_to_ground':cruise_altitude}

       return sBoom_data

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

class deformed_profile:
    '''Create and deform relative humidity profiles.

        Inputs:
         - humidity_profile: relative humidity profile from ground up through
                             atmosphere [altitude, relative humidity [%]]
         - temperature_profile: temperature profile from ground up through
                                atmosphere [altitude, temperature [C]]
    '''

    def __init__(self, humidity_profile, temperature_profile):
        self.altitudes, self.relative_humidities = package_data(humidity_profile)
        self.altitudes, self.temperatures = package_data(temperature_profile)
        self._calculate_vapor_pressures(self.relative_humidities, self.temperatures)

    def _calculate_vapor_pressures(self, humidity_profile, temperature_profile):
       '''calculate_vapor_pressures calculates the saturation_vapor_pressures
       and the actual_vapor_pressures given a temperature and humidity
       profile'''
       self.saturation_vapor_pressures = []
       for t in temperature_profile:
           if t>=0:
               self.saturation_vapor_pressures.append(0.61121*np.exp((18.678-
                                                    (t/234.5))*(t/(257.14+t))))
           else:
               self.saturation_vapor_pressures.append(0.61115*np.exp((23.036-
                                                    (t/333.7))*(t/(279.82+t))))

       self.actual_vapor_pressures = [humidity_profile[i]/100*
                                     self.saturation_vapor_pressures[i] for i in
                                     range(len(humidity_profile))]

    def _trapezoidal_area(self, x, y):
        """Compute the Riemann Sum using trapezoids."""
        areas = [(y[i-1]+y[i])*(x[i]-x[i-1])/2 for i in range(1, len(x))]
        total_area = sum(areas)
        return total_area

    def _calculate_humidity_profile(self, vapor_pressures):
        '''calculate_humidity_profile calculates the relative humidity values
        for all vapor pressure values in vapor_pressures'''
        relative_humidity_profile = [100*vapor_pressures[i]/
                                     self.saturation_vapor_pressures[i] for i in
                                     range(len(vapor_pressures))]

        return relative_humidity_profile

    def _filter_vapor_pressures(self, actual_vapor_pressures):
        '''filter_vapor_pressures makes sure that the actual_vapor_pressures
        do not fall out of range of the saturation_vapor_pressures'''
        for i in range(len(actual_vapor_pressures)):
            if actual_vapor_pressures[i] > self.saturation_vapor_pressures[i]:
                actual_vapor_pressures[i] == self.saturation_vapor_pressures[i]
            elif actual_vapor_pressures[i] < 0:
                actual_vapor_pressures[i] == 0

        return actual_vapor_pressures

    def create_spike(self, base, height, peak):
        self.deformation = deformation(base=base, height=height, peak=peak)

    def add_spike(self):
        '''add_spike adds a spike to an inputted humidity profile without
        changing the integral of the humidity profile'''
        self.spiked_vapor_profile = self.actual_vapor_pressures
        for i in range(len(self.altitudes)):
             if self.altitudes[i] > self.deformation.peak-(self.deformation.base
                            /2) and self.altitudes[i] <= self.deformation.peak:
                 self.spiked_vapor_profile[i] = (2*self.deformation.height/
                                                self.deformation.base)*\
                                                (self.altitudes[i]-
                                                (self.deformation.peak-
                                                (self.deformation.base/2)))
             elif self.altitudes[i] < self.deformation.peak+\
                            (self.deformation.base/2) and self.altitudes[i] >\
                            self.deformation.peak:
                 self.spiked_vapor_profile[i] = (2*self.deformation.height/
                                                self.deformation.base)*\
                                                (self.deformation.peak-
                                                self.altitudes[i]+\
                                                self.deformation.base/2)

        self.spiked_vapor_profile = self._filter_vapor_pressures(self.spiked_vapor_profile)

        self.spiked_rh_profile = self._calculate_humidity_profile(
                                                    self.spiked_vapor_profile)

    def change_average(self, vapor_multiplier):
        '''change_average multiplies the inputted humidity_profile by a constant
        multiplier'''
        # Area of original and spiked profiles (vapor pressure profiles)
        self.original_area = self._trapezoidal_area(self.altitudes,
                                                    self.actual_vapor_pressures)
        self.spiked_area = self._trapezoidal_area(self.altitudes,
                                                  self.spiked_vapor_profile)

        # Correction factor for area added by spike deformation
        correction_factor = self.original_area/self.spiked_area

        # Create scaled humidity profile
        scaled_vapor_profile = [self.spiked_vapor_profile[i]*vapor_multiplier*
                                correction_factor for i in
                                range(len(self.spiked_vapor_profile))]
        scaled_vapor_profile = self._filter_vapor_pressures(scaled_vapor_profile)

        self.scaled_rh_profile = self._calculate_humidity_profile(scaled_vapor_profile)

################################################################################
from optimization_tools.DOE import DOE
import time
import pickle

# Prepare sBoom_data for humidity_profile_DOE
sBoom_data = prepare_standard_profiles()

def humidity_profile_DOE(inputs):
    profile_deformed = deformed_profile(inputs['sBoom']['profiles'][2],
                                        inputs['sBoom']['profiles'][0])
    profile_deformed.create_spike(base=inputs['base'], height=inputs['height'],
                                    peak=inputs['peak'])
    profile_deformed.add_spike()
    profile_deformed.change_average(vapor_multiplier=inputs['amount_of_vapor'])

    final_profile = package_data(profile_deformed.altitudes,
                                 profile_deformed.scaled_rh_profile,
                                 method='pack')
    inputs['sBoom']['profiles'] = inputs['sBoom']['profiles'][:2] + [final_profile]

    noise = boom_runner(inputs['sBoom']['profiles'],
                        inputs['sBoom']['height_to_ground'])

    return {'noise':noise}

# Define points
problem = DOE(levels=2, driver='Full Factorial')
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
             ylabel=['PLdB'], number_y=2)

# Store DOE
date = 'test_20190510'
fileObject = open('humidity_DOE/DOE_FullFactorial_'+date, 'wb')
pickle.dump(problem, fileObject)
fileObject.close()
