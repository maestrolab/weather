if __name__ == '__main__':
    '''Script to parametrize the humidity profile using a single Hermite spline.'''

    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import differential_evolution
    from scipy.interpolate import interp1d

    from weather.boom import boom_runner, prepare_weather_sBoom

    from parametrize_humidity import ParametrizeHumidity
    from misc_humidity import package_data, convert_to_celcius,\
                              prepare_standard_profiles
    from misc_cluster import truncate_at_altitude
    from choose_cluster import choose_method

    data = pickle.load(open('./../../../72469_profiles.p','rb'))

    # Prepare standard pressure profile
    path = './../../../../../data/weather/standard_profiles/standard_profiles.p'
    standard_profiles = prepare_standard_profiles(standard_profiles_path = path)
    standard_alts, standard_pressures = package_data(standard_profiles['pressure'])
    fun_interp = interp1d(standard_alts, standard_pressures)

    days = list(range(len(data['humidity'])))

    for day in days:
        print(day)
        # Truncate profiles to cruise altitude
        humidity = truncate_at_altitude(data['humidity'][day])
        temperature = truncate_at_altitude(data['temperature'][day])

        # Parametrization process
        profile_altitudes, relative_humidities = package_data(humidity)
        profile_altitudes, temperatures = package_data(temperature)
        pressures = fun_interp(profile_altitudes)

        method, RMSE_method, bounds = choose_method(humidity)

        print(method, RMSE_method)

        p_profile = ParametrizeHumidity(profile_altitudes, relative_humidities,
                                        temperatures, pressures, bounds = bounds,
                                        geometry_type = method)

        # Optimize profile
        fun = p_profile.RMSE
        bounds_normalized = [(0,1) for i in range(len(bounds))]
        res = differential_evolution(fun, bounds = bounds_normalized,
                                     args = [RMSE_method], popsize = 100,
                                     updating = 'deferred', workers = 10)

        # Plot optimized profile
        x = p_profile.normalize_inputs(res.x)
        p_profile.geometry(x)
        p_profile.calculate_humidity_profile()
        p_profile.RMSE(res.x, profile_type = RMSE_method)
        # p_profile.plot()
        p_profile.plot(profile_type = 'relative_humidities')
        plt.show()

        # Calculate noise
        p_humidity_profile = package_data(p_profile.p_alts, p_profile.p_rhs, method='pack')
        noise = {'original':0,'parametrized':0,'difference':0}

        height_to_ground = p_profile.p_alts[-1]

        # Noise calculations (original profile)
        boom_runner_path = './../../../../../data/nearfield/25D_M16_RL5.p'
        sBoom_data = [temperature, 0, humidity]
        noise['original'] = boom_runner(sBoom_data, height_to_ground,
                                        nearfield_file=boom_runner_path)

        # Noise calculations (parametrized profile)
        sBoom_data_parametrized = list(sBoom_data[0:2]) + [p_humidity_profile]
        noise['parametrized'] = boom_runner(sBoom_data_parametrized, height_to_ground,
                                            nearfield_file=boom_runner_path)

        noise['difference'] = noise['original'] - noise['parametrized']

        print(noise)
