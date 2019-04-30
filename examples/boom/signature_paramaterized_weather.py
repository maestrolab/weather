import pickle
from weather.boom import boom_runner, process_data


def parametrize_humidity(humidity):
    """parametrize_humidity takes a humidity profile represented as a list of
    data points (each data point has a corresponding altitude stored with it)
    and returns a parametrized humidity profile. The goal of the parametrized
    humidity profile is to simplify the input to the optimization algorithm
    while mainting an accurate percieved noise level (+/- 0.1 PLdB).
    """
    ############################################################################
    # Clumps of data points are clustered together and average humidity and
    #   average altitude computed and inputted into sboom.
    # humidity_vals = [h[1] for h in humidity]
    # temp_vals = [t[0] for t in humidity]
    #
    # humidity_vals_trial = [humidity_vals[0]]
    # humidity_vals_trial.append(sum(humidity_vals[1:3])/2)
    # humidity_vals_trial.append(sum(humidity_vals[3:5])/2)
    # humidity_vals_trial.append(sum(humidity_vals[5:8])/3)
    # humidity_vals_trial.append(sum(humidity_vals[8:13])/5)
    # humidity_vals_trial.append(humidity_vals[13])
    #
    # temp_vals_trial = [temp_vals[0]]
    # temp_vals_trial.append(sum(temp_vals[1:3])/2)
    # temp_vals_trial.append(sum(temp_vals[3:5])/2)
    # temp_vals_trial.append(sum(temp_vals[5:8])/3)
    # temp_vals_trial.append(sum(temp_vals[8:13])/5)
    # temp_vals_trial.append(temp_vals[13])
    # # print(humidity_vals_trial, temp_vals_trial)

    # humidity = []
    # for i in range(len(humidity_vals_trial)):
    #     humidity.append([temp_vals_trial[i],humidity_vals_trial[i]])
    ############################################################################
    # Point added via linear interpolation between two points in profile
    #   Did not change perceived loudness --> linear interpolation used in
    #   sboom?
    # new_humidity = []
    # new_humidity = [humidity[i] for i in range(13)]
    # new_humidity.append([12750.0,67.92])
    # new_humidity.append(humidity[13])
    ############################################################################
    # Points removed if altitude was above a certain threshold
    #   Works to an extent --> does not reduce number of data points enough
    # threshold = 4000 # [ft]
    # new_humidity = [h for h in humidity if h[0] < threshold]
    ############################################################################
    # Points added separately to four corners and center of profile to study
    #   the effect each one has on the perceived loudness.
    #   Essentially looks at if certain humidity values effect perceived
    #       loudness more or less at different altitudes.
    #
    # # humidity_altitude
    # low_high = [15000.0,5.0] # humidity[-1]
    # low_low = [150.0,5.0] # humidity[1]
    # high_high = [15000.0,65.0] # humidity[-1]
    # high_low = [150.0,65.0] # humidity[1]
    #
    # new_humidity = humidity[:1]
    # new_humidity.append(high_low)
    # [new_humidity.append(h) for h in humidity[1:]]
    ############################################################################
    new_humidity = [humidity[0]]
    new_humidity.append(humidity[5])
    new_humidity.append(humidity[10])
    new_humidity.append(humidity[14])
    # new_humidity.append(humidity[15])
    new_humidity.append(humidity[-1])

    # humidity_vals = [h[1] for h in new_humidity]
    # alt_vals = [alt[0] for alt in new_humidity]

    # Plot the humidity profile to help visualize differences between
    #   parameterization methods.
    # import matplotlib.pyplot as plt

    # fig = plt.figure()
    # plt.plot(humidity_vals, alt_vals)
    # plt.xlabel('Relative Humidity')
    # plt.ylabel('Altitude [m]')  # Check if altitude is in feet or meters
    # plt.grid(True)
    #
    # filename = 'Test_1.png'
    # plt.savefig('./../../data/weather/parametrized_humidity_profiles/' +
    #             filename)

    print(new_humidity)
    return new_humidity


day = '18'
month = '06'
year = '2018'
hour = '12'
lat = 32
lon = -100
alt_ft = 45000.
alt = alt_ft * 0.3048

data, altitudes = process_data(day, month, year, hour, alt,
                               directory='../../data/weather/')
key = '%i, %i' % (lat, lon)
weather_data = data[key]

# Height to ground (HAG)
index = list(data.keys()).index(key)
height_to_ground = altitudes[index]  # In meters

data[key]['humidity'] = parametrize_humidity(data[key]['humidity'])
noise = boom_runner(data, height_to_ground, index)

print(noise)
