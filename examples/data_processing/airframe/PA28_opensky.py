from weather.scraper.flight_conditions import properties, Airframe
import pickle
import math

# Define object
fuel = 181*6.01*0.4535
initial_mass = 1090.9
final_mass = initial_mass-fuel
PA28_props = properties({'Cl_alpha': 4.762, 'Cl_0': 4.762*2.6*math.pi/180,
                         'planform': 14.86, 'density': 0.770488088,
                         'mass_min': final_mass, 'mass_max': initial_mass,
                         'incidence': 0.})
PA28 = Airframe(airframe='PA28', timestamp=1549036800,
                filepath='../../data/flight_plan/v_aoa_pickles/icao24s_',
                properties=PA28_props)
PA28.update_OpenSkyApi(desired_airframes=['PA28'], num_data_points=4000,
                        time_increment=15, save_data=False)
# PA28.retrieve_data()
# PA28.train_pdf(1000)

# # Plot PDF
# PA28.plot_pdf()

pickle.dump(PA28, open("pa28_10000.p", "wb"))