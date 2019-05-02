import pickle
from weather.boom import boom_runner
from weather.scraper.twister import process_data
import matplotlib.pyplot as plt
import numpy as np


def plot_humidity(humidity, noise, filename, save_fig=False, save_fig_filepath=None):
    """Function to plot humidity profile"""
    humidity_vals = [h[1] for h in humidity]
    alt_vals = [alt[0] for alt in humidity]

    fig = plt.figure()
    plt.plot(humidity_vals, alt_vals)
    plt.title('Perceived Loudness = %.4f' % noise)
    plt.xlabel('Relative Humidity [%]')
    #plt.xlabel('Vapor Pressure [kPa]')
    plt.ylabel('Altitude [m]')
    plt.grid(True)

    if save_fig == True:
        plt.savefig(save_fig_filepath + filename)
    # plt.show()


def trapezoidal_area(x, y):
    """Compute the Riemann Sum using trapezoids."""
    areas = [(y[i-1]+y[i])*(x[i]-x[i-1])/2 for i in range(1, len(x))]
    total_area = sum(areas)

    return total_area


def parametrize_humidity(humidity, temperature, cutoff_altitude):
    """parametrize_humidity parametrizes the humidity profile"""

    # Remove points above aircraft cruise altitude
    cruise_altitude = 13000  # [m] Add as input into function for user to control
    new_humidity = [h for h in humidity if h[0] < cruise_altitude]

    """Need to determine freezing point at different pressures."""
    # Remove points with temperatures below freezing point
    # freezing_temp = 32 # degrees F
    # new_humidity = [humidity[i] for i in range(len(humidity)) if
    #                 temperature[i][1] < freezing_temp]

    saturation_vapor_pressures = [
        [t[0], 0.61121*np.exp((18.678-(t[1]/234.5))*(t[1]/(257.14+t[1])))] for t in temperature]
    actual_vapor_pressures = [[new_humidity[i][0], new_humidity[i][1]/100 *
                               saturation_vapor_pressures[i][1]] for i in range(len(new_humidity))]

    # Compute area of vapor pressures profile.
    altitudes = [alt[0] for alt in actual_vapor_pressures]
    vapor_pressures = [p_vap[1] for p_vap in actual_vapor_pressures]
    total_area = trapezoidal_area(altitudes, vapor_pressures)

    # Represent area as a triangle (integral stays the same).
    #   1. Calculate the vapor pressure value at zero altitude that makes a
    #       triangle with the same area as the vapor pressure profile.
    equivalent_vapor_pressure = 2*total_area/cutoff_altitude

    # for svp in saturation_vapor_pressures:
    #     if svp[0] == cutoff_altitude:
    #         saturation_vapor_pressure_at_cutoff = svp[1]

    #   2. Change vapor pressure value back into relative humidity value.
    #print(saturation_vapor_pressure_at_cutoff, equivalent_vapor_pressure)
    equivalent_relative_humidity = equivalent_vapor_pressure/saturation_vapor_pressures[0][1]

    #   3. new_humidity = [[0,found_rel_humidity],[middle_altitude,0],[max_altitude,0]]
    new_humidity = [[0, equivalent_relative_humidity*100],
                    [cutoff_altitude, 0], [actual_vapor_pressures[-1][0], 0]]

    return new_humidity


day = '18'
month = '06'
year = '2018'
hour = '12_'
lat = 32
lon = -120
alt_ft = 45000.
alt = alt_ft * 0.3048

data, altitudes = process_data(day, month, year, hour, alt,
                               directory=directory='../../../data/weather/twister/',
                               convert_celcius_to_fahrenheit=True,
                               convert_to_fahrenheit=True)

key = '%i, %i' % (lat, lon)

altitude_list = [alt[0] for alt in data[key]['temperature'][1:-1]]
for altitude in altitude_list:
    data, altitudes = process_data(day, month, year, hour, alt,
                                   directory=directory='../../../data/weather/twister/',
                                   convert_celcius_to_fahrenheit=True)

    key = '%i, %i' % (lat, lon)
    weather_data = data[key]

    # Height to ground (HAG)
    index = list(data.keys()).index(key)
    height_to_ground = altitudes[index]  # In meters

    data[key]['temperature'] = [[t[0], t[1]*(9/5)+32] for t in
                                data[key]['temperature']]

    data[key]['humidity'] = parametrize_humidity(data[key]['humidity'],
                                                 data[key]['temperature'],
                                                 cutoff_altitude=altitude)

    noise = boom_runner(data, height_to_ground, index)

    filename = '%s, %s Cutoff Altitude = %.1f.png' % (lat, lon, altitude)
    save_fig_filepath = './../../../../Research_Spring 2019/parametrized_humidity_profiles/'
    plot_humidity(data[key]['humidity'], noise, filename, save_fig=True,
                  save_fig_filepath=save_fig_filepath)
