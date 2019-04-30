from weather.scraper.flight_conditions import properties, Airframe

# Define object
C172_props = properties({'Cl_alpha': 5.143, 'Cl_0': 0.31,
                         'planform': 16.1651, 'density': 0.770488088,
                         'mass_min': 618., 'mass_max': 919.,
                         'incidence': 0.})
C172 = Airframe(airframe='B737', timestamp=1549036800,
                filepath='../../data/flight_plan/v_aoa_pickles/icao24s_',
                properties=C172_props)
C172.update_OpenSkyApi(desired_airframes=['C172'], num_data_points=1000,
                        time_increment=15, save_data=False)
C172.retrieve_data()
C172.train_pdf(1000)

# Plot PDF
C172.plot_pdf()
