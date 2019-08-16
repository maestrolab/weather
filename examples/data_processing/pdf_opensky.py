from weather.scraper.flight_conditions import properties, Airframe

# Define object
fuel = 56*6.01*0.4535
initial_mass = 1111
final_mass = initial_mass-fuel
C172_props = properties({'Cl_alpha': 5.143, 'Cl_0': 0.31,
                         'planform': 16.1651, 'density': 0.770488088,
                         'mass_min': final_mass, 'mass_max': initial_mass,
                         'incidence': 0.})
C172 = Airframe(airframe='C172', timestamp=1549036800,
                filepath='../../data/flight_plan/v_aoa_pickles/icao24s_',
                properties=C172_props)
C172.update_OpenSkyApi(desired_airframes=['C172'], num_data_points=1000,
                        time_increment=15, save_data=False)
C172.retrieve_data()
C172.train_pdf(1000)

# Plot PDF
C172.plot_pdf()
