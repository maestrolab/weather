import numpy as np
from scipy.interpolate import interp1d

from misc_humidity import calculate_vapor_pressures

def interpolate_profiles(altitudes, humidity, temperature, pressure = np.array([])):
    rh_interpolated = np.zeros((len(humidity), len(altitudes), 2))
    temp_interpolated = np.zeros((len(temperature), len(altitudes), 2))
    for i in range(len(humidity)):
        alt_, rh_vals = np.array(humidity[i]).T
        alt, temp_vals = np.array(temperature[i]).T

        # FIX TO WORK FOR ANY SET OF PROFILES!!!!!!
        if len(rh_vals) != len(temp_vals):
            if len(alt) > len(alt_):
                alt = alt_[:]
                temp_vals = temp_vals[1:]
            else:
                rh_vals = rh_vals[:-1]

        rh_fun = interp1d(alt, rh_vals)
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

def normalize_inputs(x, bounds, inverse = False):
    if inverse:
        normalized_inputs = (bounds[1]-bounds[0])*x+bounds[0]
    else:
        normalized_inputs = (x-bounds[0])/(bounds[1]-bounds[0])
    return normalized_inputs

def define_bounds(profiles, type = ['min','max']):
    bounds = np.zeros((len(profiles), 2))
    for i in range(len(profiles)):
        lb, ub = [np.min(profiles[i]), np.max(profiles[i])]
        bounds[i] = [lb, ub]
        types = {'constant_min':0, 'constant_max':100,
                 'min':bounds[i][0], 'max':bounds[i][1]}
        bounds[i] = [types[type[0]], types[type[1]]]

    return bounds

def normalize_inputs_temp_append(x, n, bounds, inverse = False):
    if inverse:
        norm1 = (bounds[0][1]-bounds[0][0])*x[:n]+bounds[0][0]
        norm2 = (bounds[1][1]-bounds[1][0])*x[n:]+bounds[1][0]
    else:
        norm1 = (x[:n]-bounds[0][0])/(bounds[0][1]-bounds[0][0])
        norm2 = (x[n:]-bounds[1][0])/(bounds[1][1]-bounds[1][0])
    normalized_inputs = np.append(norm1, norm2)
    return normalized_inputs

def define_bounds_temp_append(profiles, n, type = [['min','max'],['min','max']]):
    bounds = np.zeros((len(profiles), 2, 2))
    for i in range(len(profiles)):
        lb_rh, ub_rh = [np.min(profiles[i][:n]), np.max(profiles[i][:n])]
        lb_t, ub_t = [np.min(profiles[i][n:]), np.max(profiles[i][n:])]
        bounds[i][0] = [lb_rh, ub_rh]
        bounds[i][1] = [lb_t, ub_t]
        rh_types = {'constant_min':0, 'constant_max':100,
                    'min':bounds[i][0][0], 'max':bounds[i][0][1]}
        temp_types = {'constant_min':-78, 'constant_max':62,
                      'min':bounds[i][1][0], 'max':bounds[i][1][1]}
        bounds[i][0] = [rh_types[type[0][0]], rh_types[type[0][1]]]
        bounds[i][1] = [temp_types[type[1][0]], temp_types[type[1][1]]]

    return bounds

def bounds_check(profiles, bounds):
    i = 0
    while i != len(profiles):
        if bounds[i][0][0] == bounds[i][0][1]:
            profiles = np.delete(profiles, i, 0)
            bounds = np.delete(bounds, i, 0)
        else:
            i += 1

    return profiles, bounds


def prepare_vapor_pressures(altitudes, humidity, temperature, pressure):
    vps_profiles = np.zeros((len(humidity), len(altitudes), 2))
    sat_vps_profiles = np.zeros((len(humidity), len(altitudes), 2))
    for i in range(len(humidity)):
        vps, sat_vps = calculate_vapor_pressures(humidity[i][:,1],
                                                 temperature[i][:,1],
                                                 pressure[:,1])
        vps_profiles[i] = np.array([altitudes, vps]).T
        sat_vps_profiles[i] = np.array([altitudes, sat_vps]).T

    return vps_profiles, sat_vps_profiles

# alt_interpolated = np.linspace(0,13500,n)
# rh_interpolated = np.zeros((len(rh), len(alt_interpolated), 2))
# temp_interpolated = np.zeros((len(rh), len(alt_interpolated), 2))
# for i in range(len(rh)):
#     alt, rh_vals = np.array(rh[i]).T
#     alt, temp_vals = np.array(temp[i]).T
#     rh_fun = interp1d(alt, rh_vals)
#     temp_fun = interp1d(alt, temp_vals)
#     rh_interpolated_values = rh_fun(alt_interpolated)
#     temp_interpolated_values = temp_fun(alt_interpolated)
#     rh_interpolated[i] = np.array([alt_interpolated, rh_interpolated_values]).T
#     temp_interpolated[i] = np.array([alt_interpolated, temp_interpolated_values]).T
#
# # Interpolate pressure profile
# alt, pres_vals = pres.T
# pres_fun = interp1d(alt, pres_vals)
# pres_interpolated_values = pres_fun(alt_interpolated)
# pres_interpolated = np.array([alt_interpolated, pres_interpolated_values]).T
