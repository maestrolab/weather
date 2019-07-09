import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, basinhopping

from parametrize_humidity import ParametrizeHumidity
from misc_humidity import package_data

from weather.boom import boom_runner
from weather import convert_to_fahrenheit

def compare_parametrizations(latitude, longitude, altitudes, relative_humidities,
                             temperatures, pressures, directory = './',
                             update_file = False):
    geometry_types = ['spline', 'spline_bump', 'spline_log', 'spline_bump_log']
    RMSE_types = ['vapor_pressures', 'relative_humidities']

    noise, sBoom_data = _compute_original_noise(altitudes, relative_humidities,
                                                temperatures)

    if update_file:
        filename = '[Lat, Lon] = [%s, %s]' % (latitude, longitude)
        file_handle = open(directory + filename + '.txt','w')
    for geometry_type in geometry_types:
        bounds = _select_bounds(geometry_type)
        profile = ParametrizeHumidity(altitudes, relative_humidities, temperatures,
                                      pressures, bounds = bounds,
                                      geometry_type = geometry_type)

        bounds_normalized = [(0,1) for i in range(len(bounds))]

        if update_file:
            file_handle.write(geometry_type+'\n')

        for RMSE_type in RMSE_types:
            # Optimize parameters for chosen geometry_type and RMSE_type
            fun = profile.RMSE
            res = differential_evolution(fun, bounds = bounds_normalized,
                                         args = [RMSE_type], popsize = 100,
                                         updating = 'deferred', workers = 10)

            x = profile.normalize_inputs(res.x)
            profile.geometry(x)
            profile.calculate_humidity_profile()
            profile.RMSE(res.x, profile_type = RMSE_type)

            noise = _compute_parametrized_noise(profile, noise, sBoom_data)

            noise['difference'] = noise['original'] - noise['parametrized']
            print(geometry_type)
            print(RMSE_type)
            print(noise['difference'])
            if update_file:
                # Write results to text file
                write_RMSE_type = '\tRMSE Type: %s\n' % (RMSE_type)
                write_RMSE = '\t\t%.4f\n' % (profile.rmse)
                write_PLdB = '\t\t%.4f\n' % (noise['difference'])
                file_handle.write(write_RMSE_type + write_RMSE + write_PLdB)

    if update_file:
        file_handle.close()

def _select_bounds(geometry_type):
    '''bounds found through trial and error of multiple runs; they can be
    adjusted if problems arise'''
    if geometry_type == 'spline':
        # bounds = [p0, p1, m0, m1, b]
        bounds = [[1.,4.], [0, 0.001], [-0.1,0.], [-0.1,0.], [8000,16000]]
    elif geometry_type == 'spline_bump':
        # bounds = [p0, p1, m0, m1, m, x, y, b]
        bounds = [[0.5,3.], [0.,0.5], [-0.1,0.], [0.,0.], [-0.1,0.], [2000,12000], \
                  [0.,0.5], [8000,18000]]
    elif geometry_type == 'spline_log':
        # bounds = [p0, p1, m0, m1, b, a]
        bounds = [[0.5,3],[0.,0.01],[-1.,1.],[-0.01,0.],[6,14],[0,7]]
    elif geometry_type == 'spline_bump_log':
        # bounds = [p0, p1, m0, m1, m, x, y, a, b]
        bounds = [[0.5,3.], [0.,0.5], [-0.1,0.], [0.,0.], [-0.1,0.], [7,10], \
                  [0.,0.5], [0,7], [8,12]]

    return bounds

def _compute_original_noise(altitudes, relative_humidities, temperatures):
    noise = {'original':0,'parametrized':0,'difference':0}
    temperatures = convert_to_fahrenheit(temperatures)
    temperature_profile = package_data(altitudes, temperatures, method = 'pack')
    humidity_profile = package_data(altitudes, relative_humidities, method = 'pack')

    nearfield_file='../../../../data/nearfield/25D_M16_RL5.p'
    sBoom_data = [temperature_profile, 0, humidity_profile]
    height_to_ground = altitudes[-1] / 0.3048
    noise['original'] = boom_runner(sBoom_data, height_to_ground,
                                    nearfield_file = nearfield_file)

    return noise, sBoom_data

def _compute_parametrized_noise(profile, noise, sBoom_data):
    parametrized_humidity_profile = package_data(profile.p_alts, profile.p_rhs,
                                                 method = 'pack')
    height_to_ground = profile.p_alts[-1] / 0.3048
    nearfield_file='../../../../data/nearfield/25D_M16_RL5.p'

    # Noise calculations (parametrized profile)
    sBoom_data_parametrized = list(sBoom_data[0:2]) +\
                              [parametrized_humidity_profile]
    noise['parametrized'] = boom_runner(sBoom_data_parametrized,
                                        height_to_ground,
                                        nearfield_file = nearfield_file)

    return noise
