import pickle
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
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

def calculate_vapor_pressures(humidity_profile, temperature_profile):
   '''calculate_vapor_pressures calculates the saturation_vapor_pressures
   and the actual_vapor_pressures given a temperature and humidity
   profile'''
   saturation_vapor_pressures = []
   for t in temperature_profile:
       if t>=0:
           saturation_vapor_pressures.append(0.61121*np.exp((18.678-
                                                (t/234.5))*(t/(257.14+t))))
       else:
           saturation_vapor_pressures.append(0.61115*np.exp((23.036-
                                                (t/333.7))*(t/(279.82+t))))

   actual_vapor_pressures = [humidity_profile[i]/100*
                                 saturation_vapor_pressures[i] for i in
                                 range(len(humidity_profile))]
   return(actual_vapor_pressures, saturation_vapor_pressures)

def calculate_humidity_profile(self, vapor_pressures, saturation_vapor_pressures):
    '''calculate_humidity_profile calculates the relative humidity values
    for all vapor pressure values in vapor_pressures'''
    relative_humidity_profile = [100*vapor_pressures[i]/
                                 saturation_vapor_pressures[i] for i in
                                 range(len(vapor_pressures))]

    return relative_humidity_profile

def cubic_parametrize(humidity_profile, temperature_profile, number_of_points=5, max_altitude=13000):
    '''cubic_parametrize'''
    altitudes, relative_humidities = package_data(humidity_profile)
    altitudes, temperatures = package_data(temperature_profile)

    actual_vps, satuation_vps = calculate_vapor_pressures(relative_humidities,
                                                          temperatures)
    d = max_altitude
    n = actual_vps[0]
    area_vps = simps(actual_vps, altitudes)

    a = np.array([[n**3, n**2, n],[6*n, 2, 0],[n**4/4, n**3/3, n**2/2]])
    b = np.array([-d, 0, area_vps-d*n])
    c = np.linalg.solve(a,b)

    parametrized_profile = {'altitudes':[], 'vapor_pressures':[]}
    parametrized_profile['vapor_pressures'] = np.arange(0,n,n/number_of_points)
    parametrized_profile['altitudes'] = [c[0]*x**3+c[1]*x**2+c[0]*x+d for x in
                                         parametrized_profile['vapor_pressures']]

    return parametrized_profile

def quadratic_parametrize(humidity_profile, temperature_profile, number_of_points=5, max_altitude=13000):
    '''quadratic_parametrize'''
    altitudes, relative_humidities = package_data(humidity_profile)
    altitudes, temperatures = package_data(temperature_profile)

    actual_vps, satuation_vps = calculate_vapor_pressures(relative_humidities,
                                                          temperatures)
    d = max_altitude
    n = actual_vps[0]
    area_vps = simps(actual_vps, altitudes)

    a = np.array([[n**2, n], [n**3/3, n**2/2]])
    b = np.array([-d, area_vps-d*n])
    c = np.linalg.solve(a,b)

    parametrized_profile = {'altitudes':[], 'vapor_pressures':[]}
    parametrized_profile['vapor_pressures'] = np.arange(0,n,n/number_of_points)
    parametrized_profile['altitudes'] = [c[0]*x**2+c[1]*x+d for x in
                                         parametrized_profile['vapor_pressures']]

    parametrized_profile['relative_humidities'] = calculate_humidity_profile(
                                                parametrized_profile['vapor_pressures'],
                                                ***Need to interpolate saturation_vps***
    )

    return parametrized_profile

################################################################################
# Prepare standard profiles to parametrize
standard_profiles = prepare_standard_profiles()

parametrized_profile = quadratic_parametrize(standard_profiles['profiles'][2],
                                         standard_profiles['profiles'][0])

fig = plt.figure()
plt.plot(parametrized_profile['vapor_pressures'], parametrized_profile['altitudes'])
plt.show()

temperatures = package_data()
height_to_ground = 13000/0.3048
noise = boom_runner(sBoom_data,height_to_ground)
