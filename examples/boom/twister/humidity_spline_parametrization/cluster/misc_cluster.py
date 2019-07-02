import numpy as np
from scipy.interpolate import interp1d
from misc_humidity import package_data, calculate_vapor_pressures

def prepare_profiles_GMM(profiles, reference_profile, cruise_altitude,
                         include_reference = False):
    interpolated_profiles = interpolate_profiles(reference_profile, profiles,
                                                 include_reference)
    initial_profile = truncate_at_altitude(interpolated_profiles[0],
                                           cruise_altitude)
    truncated_profiles = np.array([initial_profile])

    for profile in interpolated_profiles[1:]:
        truncated_profile = truncate_at_altitude(profile, cruise_altitude)
        truncated_profiles = np.append(truncated_profiles, [truncated_profile],
                                       axis = 0)

    return truncated_profiles

def interpolate_profiles(reference_profile, profiles, include_reference = False):
    '''
        Inputs:
         - reference_profile: single profile in the form [[alt1,rh1],[alt2,rh2],...]
         - profiles: list of profiles in the form [[list1],[list2],...]

         Returns:
          - interpolated_profiles: list of profiles in the form [[list1],[list2],...]
    '''
    ref_alts, ref_rh = package_data(reference_profile)
    ref_alts = np.array(ref_alts)

    # Interpolate each profile
    interpolated_profiles = [reference_profile]

    for profile in profiles:
        alts, rh = package_data(profile)
        alts = np.array(alts)

        fun = interp1d(alts, rh)

        # Use altitudes below the maximum altitude in the profile
        if ref_alts[-1] > alts[-1]:
            index = np.where(ref_alts > alts[-1])
            alts_to_interp = ref_alts[:index[0][0]]
        else:
            alts_to_interp = ref_alts[:]

        interp_rh = fun(alts_to_interp)

        interpolated_profile = package_data(alts_to_interp, interp_rh,
                                            method = 'pack')

        interpolated_profiles = interpolated_profiles + [interpolated_profile]

    if not include_reference:
        interpolated_profiles = interpolated_profiles[1:]

    return interpolated_profiles

def profile_mixture_vector(profiles):
    initial_alts, initial_rh = package_data(profiles[0])
    mixture_vector = np.array([initial_rh])

    # Organize mixture vector for model input
    for p in profiles[1:]:
        alts, rh = package_data(p)
        rh = np.array(rh)
        mixture_vector = np.append(mixture_vector, [rh], axis = 0)

    return mixture_vector

def truncate_at_altitude(profile, cruise_altitude = 16500):
    alts, rh = package_data(profile)
    alts = np.array(alts)

    if (alts[-1] > cruise_altitude).any():
        index = np.where(alts > cruise_altitude)

        alts = alts[:index[0][0]]
        rh = rh[:index[0][0]]

    truncated_profile = package_data(alts, rh, method = 'pack')

    return np.array(truncated_profile)

def predict_clusters(profiles, model):
    cluster_assignments = {i:0. for i in range(len(profiles))}

    for i in range(len(profiles)):
        alts, vals = package_data(profiles[i])
        vals = np.array(vals)
        cluster = model.predict([vals])

        cluster_assignments[i] = cluster

    return cluster_assignments

def calculate_average_profile(profiles, clusters, model):
    n_clusters = len(model.weights_)
    count = {i:0 for i in range(n_clusters)}
    empty_profile = np.array([0. for i in range(len(profiles[0]))])
    average_profile = {i:empty_profile for i in range(n_clusters)}

    keys = clusters.keys()
    for key in keys:
        alts, vals = package_data(profiles[key])
        count[clusters[key][0]] += 1
        average_profile[clusters[key][0]] = average_profile[clusters[key][0]] +\
                                         np.array(vals)

    for n in range(n_clusters):
        average_profile[n] = average_profile[n]/count[n]

    return average_profile

def vapor_pressures_GMM(humidity_profiles, temperature_profiles,
                        pressure_profiles):
    alts, rh = package_data(humidity_profiles[0])
    alts, temp = package_data(temperature_profiles[0])
    alts, pres = package_data(pressure_profiles[0])
    initial_vps = calculate_vapor_pressures(rh, temp, pres)
    initial_vps_profile = package_data(alts, initial_vps[0], method = 'pack')
    vps_profiles = np.array([initial_vps_profile])
    for i in range(1,len(humidity_profiles)):
        alts, rh = package_data(humidity_profiles[i])
        alts, temp = package_data(temperature_profiles[i])
        alts, pres = package_data(pressure_profiles[i])
        vps = calculate_vapor_pressures(rh, temp, pres)
        vps_profile = package_data(alts, vps[0], method = 'pack')
        vps_profiles = np.append(vps_profiles, [vps_profile], axis = 0)

    return vps_profiles
