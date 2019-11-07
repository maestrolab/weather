import numpy as np
from scipy.interpolate import interp1d

def profile_bounds(profiles):
    bounds = np.zeros((len(profiles), 2))
    for i in range(len(profiles)):
        bounds[i][0], bounds[i][1] = [np.min(profiles[i]), np.max(profiles[i])]

    return bounds

def normalize_profile(profile, type, inverse = False):
    if type == 'relative_humidities':
        bounds = [0,100]
    elif type == 'temperatures':
        bounds = [-100,90]

    if inverse:
        norm = (bounds[1]-bounds[0])*(profile)+bounds[0]
    else:
        norm = (profile-bounds[0])/(bounds[1]-bounds[0])

    return norm

def bounds_check_profile(profiles, bounds):
    i = 0
    while i != len(profiles):
        if bounds[i][0] == bounds[i][1]:
            profiles = np.delete(profiles, i, 0)
            bounds = np.delete(bounds, i, 0)
        elif (profiles[i][0] == profiles[i][74]) and \
             (profiles[i][0] == profiles[i][40]):
             profiles = np.delete(profiles, i, 0)
             bounds = np.delete(bounds, i, 0)
        else:
            i += 1

    return profiles, bounds

def normalize_bounds(bounds, type, inverse = False):
    if type == 'relative_humidities':
        c_bounds = [0,100]
    elif type == 'temperatures':
        c_bounds = [-100,90]

    if inverse:
        norm = (c_bounds[1]-c_bounds[0])*bounds+c_bounds[0]
    else:
        norm = (bounds-c_bounds[0])/(c_bounds[1]-c_bounds[0])

    return norm

def NN_normalize(profile, bounds, inverse = False):
    if inverse:
        norm = (bounds[1]-bounds[0])*profile+bounds[0]
    else:
        norm = (profile-bounds[0])/(bounds[1]-bounds[0])

    return norm
