import numpy as np
from scipy.interpolate import interp1d

from misc_humidity import calculate_vapor_pressures

def interpolate_profiles(altitudes, humidity, temperature, pressure = np.array([])):
    rh_interpolated = np.zeros((len(humidity), len(altitudes), 2))
    temp_interpolated = np.zeros((len(temperature), len(altitudes), 2))
    for i in range(len(humidity)):
        alt_, rh_vals = np.array(humidity[i]).T
        alt, temp_vals = np.array(temperature[i]).T
        rh_fun = interp1d(alt_, rh_vals)
        temp_fun = interp1d(alt, temp_vals)
        rh_interpolated_values = rh_fun(altitudes)
        temp_interpolated_values = temp_fun(altitudes)
        rh_interpolated[i] = np.array([altitudes, rh_interpolated_values]).T
        temp_interpolated[i] = np.array([altitudes, temp_interpolated_values]).T

    if pressure.any():
        # Interpolate pressure profile
        alt, pres_vals = pressure.T
        pres_fun = interp1d(alt, pres_vals)
        pres_interpolated_values = pres_fun(altitudes)
        pres_interpolated = np.array([altitudes, pres_interpolated_values]).T

        return rh_interpolated, temp_interpolated, pres_interpolated
    else:
        return rh_interpolated, temp_interpolated

def normalize_inputs(x, n, bounds, elevation_at_ground = np.array([]),
                               inverse = False):
    if inverse:
        norm1 = (bounds[0][1]-bounds[0][0])*x[:n]+bounds[0][0]
        norm2 = (bounds[1][1]-bounds[1][0])*x[n:]+bounds[1][0]
        if elevation_at_ground.any():
            norm_ground = (bounds[2][1]-bounds[2][0])*\
                          elevation_at_ground+bounds[2][0]
    else:
        norm1 = (x[:n]-bounds[0][0])/(bounds[0][1]-bounds[0][0])
        norm2 = (x[n:]-bounds[1][0])/(bounds[1][1]-bounds[1][0])
        if elevation_at_ground.any():
            norm_ground = (elevation_at_ground-bounds[2][0])/\
                          (bounds[2][1]-bounds[2][0])

    normalized_inputs = np.append(norm1, norm2)

    if elevation_at_ground.any():
        normalized_inputs = np.append(normalized_inputs, norm_ground)

    return normalized_inputs

def define_bounds(profiles, n, elevation_bounds = [-1000,13500],
                            type = [['min','max'],['min','max']]):
    bounds = np.zeros((len(profiles), 3, 2))
    for i in range(len(profiles)):
        lb_rh, ub_rh = [np.min(profiles[i][:n]), np.max(profiles[i][:n])]
        lb_t, ub_t = [np.min(profiles[i][n:]), np.max(profiles[i][n:])]
        bounds[i][0] = [lb_rh, ub_rh]
        bounds[i][1] = [lb_t, ub_t]
        rh_types = {'constant_min':2.162, 'constant_max':74.304,
                    'min':bounds[i][0][0], 'max':bounds[i][0][1]}
        temp_types = {'constant_min':-71.915, 'constant_max':52.803,
                      'min':bounds[i][1][0], 'max':bounds[i][1][1]}
        bounds[i][0] = [rh_types[type[0][0]], rh_types[type[0][1]]]
        bounds[i][1] = [temp_types[type[1][0]], temp_types[type[1][1]]]
        bounds[i][2] = [elevation_bounds[0], elevation_bounds[1]]

    return bounds

def bounds_check(profiles, bounds, max_ground_elevation = 5000, index = False):
    '''max_ground_elevation included to allow user to set a limit on what the
    highest ground elevation should be (based off of region of interest)'''
    i = 0
    stored_index = None
    while i != len(profiles):
        if bounds[i][0][0] == bounds[i][0][1]:
            profiles = np.delete(profiles, i, 0)
            bounds = np.delete(bounds, i, 0)
            if index:
                stored_index = i
        elif bounds[i][0][0] > max_ground_elevation:
            profiles = np.delete(profiles, i, 0)
            bounds = np.delete(bounds, i, 0)
            if index:
                stored_index = i
        elif (profiles[i][0] == profiles[i][75]) and \
             (profiles[i][0] == profiles[i][40]):
            profiles = np.delete(profiles, i, 0)
            bounds = np.delete(bounds, i, 0)
            if index:
                stored_index = i
        else:
            i += 1

    if index:
        return profiles, bounds, stored_index
    else:
        return profiles, bounds
