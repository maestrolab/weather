if __name__ == "__main__":
    '''Script to parametrize the humidity profile using the following four
    parametrization techniques:
     - single spline
     - two splines
     - single spline in log(altitude) domain
     - two splines in log(altitude) domain
    '''
    from weather.scraper.twister import process_data
    from misc_humidity import package_data, convert_to_celcius
    from compare_parametrizations import compare_parametrizations

    day = '18'
    month = '06'
    year = '2018'
    hour = '12_'
    alt_ft = 45000.
    alt = alt_ft * 0.3048

    data, altitudes = process_data(day, month, year, hour, alt,
                                   directory='../../../../data/weather/twister/',
                                   convert_celcius_to_fahrenheit=True)

    latitudes = list(range(41,45))
    longitudes = list(range(-120,-115))

    for lat in latitudes:
        for lon in longitudes:
            key = '%i, %i' % (lat, lon)
            weather_data = data[key]
            print(key)
            # Parametrization process
            profile_altitudes, relative_humidities = package_data(weather_data['humidity'])
            profile_altitudes, temperatures = package_data(weather_data['temperature'])
            profile_altitudes, pressures = package_data(weather_data['pressure'])
            temperatures = convert_to_celcius(temperatures)

            compare_parametrizations(lat, lon, profile_altitudes, relative_humidities,
                                     temperatures, pressures,
                                     directory = './compare_parametrization_methods/',
                                     update_file = True)
