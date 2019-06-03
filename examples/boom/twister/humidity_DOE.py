import pickle
import numpy as np
from weather.boom import boom_runner
from scipy.integrate import simps

def package_data(data1, data2=None, method='unpack'):
    '''package_data packs or unpacks data in the form [[data1, data2]]'''
    if method == 'pack':
        packed_data = [[data1[i],data2[i]] for i in range(len(data1))]
        return packed_data
    elif method == 'unpack':
        unpacked_data_1 = [d[0] for d in data1]
        unpacked_data_2 = [d[1] for d in data1]
        return unpacked_data_1, unpacked_data_2

def prepare_standard_profiles(cruise_altitude=13000, standard_profiles_path=
'./../../../data/weather/standard_profiles/standard_profiles.p'):
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

class Deformation:
    '''Create a deformation to add to humidity profiles.

        Inputs:
         - base: base of deformation [m]
         - height: magnitude of desired deformation [kPa or %]
         - peak: altitude value at which to add deformation [m]
    '''

    def __init__(self, base, height, peak):
        self.base = base
        self.height = height
        self.peak = peak

class DeformedProfile:
    '''Create and deform relative humidity profiles.

        Inputs:
         - humidity_profile: relative humidity profile from ground up through
                             atmosphere [altitude, relative humidity [%]]
         - temperature_profile: temperature profile from ground up through
                                atmosphere [altitude, temperature [C]]
         - profile_type: desired profile type for DOE study (determines what
                         profiles to perform the DOE on)
    '''

    def __init__(self, humidity_profile, temperature_profile, profile_type):
        self.altitudes, self.relative_humidities = package_data(humidity_profile)
        self.altitudes, self.temperatures = package_data(temperature_profile)
        self._calculate_vapor_pressures(self.relative_humidities, self.temperatures)
        self.profile_type = profile_type

    def _calculate_vapor_pressures(self, humidity_profile, temperature_profile):
       '''_calculate_vapor_pressures calculates the saturation_vapor_pressures
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

    def _calculate_humidity_profile(self, vapor_pressures):
        '''_calculate_humidity_profile calculates the relative humidity values
        for all vapor pressure values in vapor_pressures'''
        relative_humidity_profile = [100*vapor_pressures[i]/
                                     self.saturation_vapor_pressures[i] for i in
                                     range(len(vapor_pressures))]

        return relative_humidity_profile

    def create_spike(self, base, height, peak):
        self.deformation = Deformation(base=base, height=height, peak=peak)

    def add_spike(self):
        '''add_spike adds a spike to an inputted humidity profile without
        changing the integral of the humidity profile'''
        if self.profile_type == 'vapor_pressures':
            spiked_profile = self.actual_vapor_pressures[:]
        elif self.profile_type == 'relative_humidities':
            spiked_profile = self.relative_humidities[:]

        # Add spike to profile
        for i in range(len(self.altitudes)):
             if self.altitudes[i] > self.deformation.peak-(self.deformation.base
                            /2) and self.altitudes[i] <= self.deformation.peak:
                 spiked_profile[i] = (2*self.deformation.height/
                                      self.deformation.base)*\
                                      (self.altitudes[i]-
                                      (self.deformation.peak-
                                      (self.deformation.base/2)))
             elif self.altitudes[i] < self.deformation.peak+\
                            (self.deformation.base/2) and self.altitudes[i] >\
                            self.deformation.peak:
                 spiked_profile[i] = (2*self.deformation.height/
                                      self.deformation.base)*\
                                      (self.deformation.peak-
                                      self.altitudes[i]+\
                                      self.deformation.base/2)

        if self.profile_type == 'vapor_pressures':
            self.spiked_vapor_profile = spiked_profile
            self.spiked_rh_profile = self._calculate_humidity_profile(spiked_profile)
        elif self.profile_type == 'relative_humidities':
            self.spiked_rh_profile = spiked_profile

    def change_average(self, multiplier=1):
        '''change_average multiplies the inputted humidity_profile by a constant
        multiplier'''
        if self.profile_type == 'vapor_pressures':
            original_profile = self.actual_vapor_pressures[:]
            spiked_profile = self.spiked_vapor_profile[:]
        elif self.profile_type == 'relative_humidities':
            original_profile = self.relative_humidities[:]
            spiked_profile = self.spiked_rh_profile[:]

        # Area of original and spiked profiles
        self.original_area = simps(original_profile, self.altitudes)
        self.spiked_area = simps(spiked_profile, self.altitudes)

        # Correction factor for area added by spike deformation
        self._correction_factor = self.original_area/self.spiked_area

        # Create scaled humidity profile
        if self.profile_type == 'vapor_pressures':
            self.scaled_vapor_profile = [self.spiked_vapor_profile[i]*multiplier*
                                         self._correction_factor for i in
                                         range(len(self.spiked_vapor_profile))]
            self.scaled_rh_profile = self._calculate_humidity_profile(self.scaled_vapor_profile)
        elif self.profile_type == 'relative_humidities':
            self.scaled_rh_profile = [self.spiked_rh_profile[i]*multiplier*
                                      self._correction_factor for i in range(
                                      len(self.spiked_rh_profile))]
            # Modify calculate vapor pressure function to use here
            self.scaled_vapor_profile = [self.scaled_rh_profile[i]/100*
                                         self.saturation_vapor_pressures[i] for i in
                                         range(len(self.scaled_rh_profile))]

    def filter_profile(self, filter_limit=1000, err=0.0001):
        '''filter_profile filters the vapor pressure profile to ensure that each
        point does not exceed the saturation vapor pressure'''
        actual_vapor_pressures = self.scaled_vapor_profile[:]

        # The profile is bounded by the saturation vapor pressures
        out_of_bounds = True
        runs = 0
        while out_of_bounds and runs < filter_limit:
            original_area = simps(actual_vapor_pressures, self.altitudes)
            out_of_bounds = False
            for i in range(len(actual_vapor_pressures)):
                if actual_vapor_pressures[i] > self.saturation_vapor_pressures[i]:
                    diff = abs(actual_vapor_pressures[i]-
                               self.saturation_vapor_pressures[i])
                    actual_vapor_pressures[i] = self.saturation_vapor_pressures[i]
                    if diff > err:
                        out_of_bounds = True
                elif actual_vapor_pressures[i] < 0:
                    actual_vapor_pressures[i] = 0
                    out_of_bounds = True
            filtered_area = simps(actual_vapor_pressures, self.altitudes)

            # Correct for the filtered out area
            filtered_correction_factor = original_area/filtered_area
            actual_vapor_pressures = [actual_vapor_pressures[i]*\
                                      filtered_correction_factor for i in
                                      range(len(actual_vapor_pressures))]
            runs += 1
            print(runs)

        self.filtered_vapor_profile = actual_vapor_pressures
        self.filtered_rh_profile = self._calculate_humidity_profile(
                                                    self.filtered_vapor_profile)

################################################################################
from optimization_tools.DOE import DOE
import time
import pickle

# Prepare sBoom_data for humidity_profile_DOE and initialize profile types
sBoom_data = prepare_standard_profiles()
profile_type = ['vapor_pressures','relative_humidities']

def humidity_profile_DOE(inputs):
    profile_deformed = DeformedProfile(inputs['sBoom']['profiles'][2],
                                        inputs['sBoom']['profiles'][0],
                                        profile_type=inputs['profile_type'])
    profile_deformed.create_spike(base=inputs['base'], height=inputs['height'],
                                    peak=inputs['peak'])
    profile_deformed.add_spike()
    profile_deformed.change_average(multiplier=inputs['amount_of_vapor'])
    profile_deformed.filter_profile(filter_limit=100000, err=0.00001)
    final_profile = package_data(profile_deformed.altitudes,
                                 profile_deformed.filtered_rh_profile,
                                 method='pack')
    updated_sBoom = {'profiles':inputs['sBoom']['profiles'][:2]+[final_profile],
                     'height_to_ground':inputs['sBoom']['height_to_ground']}

    noise = boom_runner(updated_sBoom['profiles'], updated_sBoom['height_to_ground'])

    return {'noise':noise}

# Define points
problem = DOE(levels=4, driver='Full Factorial')
problem.add_variable('amount_of_vapor', lower=0.9, upper=1.1, type=float) # [%]
problem.add_variable('base', lower=100, upper=3000, type=float) # [m]
problem.add_variable('height', lower=2, upper=90, type=float) # [kPa]
problem.add_variable('peak', lower=1000, upper=9000, type=float) # [m]
problem.define_points()

# Run for a function with dictionary as inputs
problem.run(humidity_profile_DOE, cte_input={'sBoom': sBoom_data,
                                             'profile_type':profile_type[1]})
problem.find_influences(not_zero=True)
problem.find_nadir_utopic(not_zero=True)
print('Nadir: ', problem.nadir)
print('Utopic: ', problem.utopic)

# Plot factor effects
problem.plot(xlabel=['amount_of_vapor', 'base', 'height', 'peak'],
             ylabel=['PLdB'], number_y=1)

# Store DOE
date = 'rh_4_levels_20190510'
fileObject = open('humidity_DOE/peak_not_0/DOE_FullFactorial_'+date, 'wb')
pickle.dump(problem, fileObject)
fileObject.close()
