import numpy as np
from scipy.interpolate import interp1d

################################################################################
#                       Interpolate profiles
################################################################################
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

################################################################################
#        Remove outliers (set values chosen based off of visual inspection)
################################################################################
def outliers(relative_humidities, temperatures, elevations, max_alt = 5000,
             min_temp_low = {'loc':40, 'val':-58},
             min_temp_high = {'loc':65, 'val':-110},
             max_temp_high = {'loc':74, 'val':-45}):
    # Prepare data for index deletion
    n = relative_humidities.shape[1]
    database = {'alt_rh':np.zeros((len(relative_humidities), n)),
                'rh':np.zeros((len(relative_humidities), n)),
                'alt_temp':np.zeros((len(temperatures), n)),
                'temp':np.zeros((len(temperatures), n)), 'elevations':elevations}
    keys = database.keys()
    for i in range(len(relative_humidities)):
        database['alt_rh'][i], database['rh'][i] = np.array(relative_humidities[i]).T
        database['alt_temp'][i], database['temp'][i] = np.array(temperatures[i]).T

    # Delete indices that violate the boundaries
    i = 0
    while i != len(database['rh']):
        delete = False
        if (np.min(database['alt_rh'][i]) > max_alt) or \
           (np.min(database['alt_temp'][i]) > max_alt):
            delete = True
        elif np.min(database['temp'][i][:min_temp_low['loc']]) < min_temp_low['val']:
            delete = True
        elif np.min(database['temp'][i][min_temp_high['loc']:]) < min_temp_high['val']:
            delete = True
        elif np.min(database['temp'][i][max_temp_high['loc']:]) > max_temp_high['val']:
            delete = True

        if delete:
            for key in keys:
                database[key] = np.delete(database[key], i, axis = 0)
        else:
            i += 1

    # Prepare data to return
    return_rh = np.zeros((len(database['alt_rh']),n,2))
    return_temp = np.zeros((len(database['alt_temp']),n,2))
    for i in range(len(return_rh)):
        return_rh[i] = np.array([database['alt_rh'][i],database['rh'][i]]).T
        return_temp[i] = np.array([database['alt_temp'][i],database['temp'][i]]).T
    elevations = np.array(database['elevations'])[:]

    return return_rh, return_temp, elevations

################################################################################
#                  Normalize inputs (using varying bounds)
################################################################################
def normalize_variable_bounds(x, n, variable_bounds,
                              elevation_at_ground = np.array([]), inverse = False):

    if inverse:
        norm1 = np.zeros((len(x[:n]),))
        for i in range(len(norm1)):
            norm1[i] = (variable_bounds['rh'][i][1]-\
                        variable_bounds['rh'][i][0])*x[i]+\
                        variable_bounds['rh'][i][0]
        norm2 = np.zeros((len(x[n:]),))
        for i in range(len(norm2)):
            norm2[i] = (variable_bounds['temp'][i][1]-\
                        variable_bounds['temp'][i][0])*x[n+i]+\
                        variable_bounds['temp'][i][0]
        if elevation_at_ground.any():
            norm_ground = (13500-(-1000))*elevation_at_ground+(-1000)

    else:
        norm1 = np.zeros((len(x[:n]),))
        for i in range(len(norm1)):
            norm1[i] = (x[i]-variable_bounds['rh'][i][0])/\
                       (variable_bounds['rh'][i][1]-\
                        variable_bounds['rh'][i][0])
        norm2 = np.zeros((len(x[n:]),))
        for i in range(len(norm2)):
            norm2[i] = (x[n+i]-variable_bounds['temp'][i][0])/\
                       (variable_bounds['temp'][i][1]-\
                        variable_bounds['temp'][i][0])
        if elevation_at_ground.any():
            norm_ground = (elevation_at_ground-(-1000))/(13500-(-1000))

    normalized_inputs = np.append(norm1, norm2)
    if elevation_at_ground.any():
        normalized_inputs = np.append(normalized_inputs, norm_ground)

    return normalized_inputs
