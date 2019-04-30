from weather.scraper.flight_conditions import properties, Airframe

# Define object
C172_props = properties({'Cl_alpha': 5.143, 'Cl_0': 0.31,
                         'planform': 16.1651, 'density': 0.770488088,
                         'mass_min': 618., 'mass_max': 919.,
                         'incidence': 0.})
C172 = Airframe(airframe='C172', timestamp=1549036800,
                filepath='../../data/flight_plan/v_aoa_pickles/icao24s_',
                csv_filepath='../../data/flight_plan/aircraftDatabases/' +
                'aircraftDatabase.csv', properties=C172_props)

# Pull and save data from OpenSky Network
C172.update_OpenSkyApi(desired_airframes=['C172'], num_data_points=1000,
                        time_increment=15, save_data=True)
