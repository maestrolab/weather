import pickle
from opensky_api import OpenSkyApi
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from weather.filehandling import output_reader
import time


class Airframe(object):
    """as"""

    def __init__(self, timestamp=1549729462, typecode='B737'):
        self.typecode = typecode  # --> (in methods, only pull [0:3])
        self.timestamp = timestamp  # Included to tell what csv to use and when to pull OpenSkyApi data from
        self.velocity = np.array([])
        self.vertrate = np.array([])
        self.angleOfAttack = np.array([])
        # + str(self.timestamp) + '_test.csv'
        self.csv_filename = 'aircraftDatabase_1549729462_test.csv'
        self.plot_filename = 'weight_pdf_v_AoA_' + self.typecode + '.png'
        self.scatter_filename = 'W=pertubations_v_AoA_scatter_' + self.typecode + '.png'
        self.filepath = '../../data/flight_plan/v_aoa_pickles/icao24s_'
        self.airframe = 'C172'

    def update_csv(self):
        """
        Instructions how to format aircraftDatabase.csv downloaded from
           OpenSky-Network.
        """
        pass  # Will include description of how to format csv for proper use

    def read_csv(self, filename):
        """Read aircraft database csv."""

        type_structure = []
        for n in range(3):
            type_structure.append('string')

        data = output_reader('../../data/flight_plan/aircraftDatabases/' + filename, separator=[','],
                             type_structure=type_structure)
        return data

    def update_icao24s(self):
        """Update list of icao24s for all typecodes.
        Generates icao24s_self.timestamp.p pickle (all icao24s sorted by typecode)
        """

        # Made separate function in case a different filename
        data = self.read_csv(self.csv_filename)
        # is wanted to be used. (type(data) == dictionary)

        typecodeDict = {'B737': {}, 'B747': {}, 'B757': {}, 'B767': {}, 'B777': {}, 'B787': {},
                        'A310': {}, 'A318': {}, 'A319': {}, 'A320': {}, 'A321': {},
                        'A330': {}, 'A340': {}, 'A350': {}, 'A380': {}, 'C172': {}, 'C180': {},
                        'C182': {}}
        typecodeKeys = list(typecodeDict.keys())
        typecodeDict = {key: {'icao24List': [], 'manufacturer': ''} for key in typecodeDict}

        # Add manufacturer information to each typecode.
        for key in typecodeKeys:
            if key[0:1] == 'B':
                typecodeDict[key]['manufacturer'] = 'Boeing'
            if key[0:1] == 'A':
                typecodeDict[key]['manufacturer'] = 'Airbus'
            if key[0:1] == 'C':
                typecodeDict[key]['manufacturer'] = 'Cessna'

        # Use typecode to gather icao24s for Boeing and Airbus aircraft
        for key in typecodeKeys[0:15]:
            i = 0  # here to offset icao24 row pull
            icao24List = []
            for typecode in data['typecode']:
                if typecode[0:3] == key[0:3]:
                    icao24List.append(data['icao24'][i+1])  # i+1: lines up icao24 w/ typecode
                i += 1
            typecodeDict[key]['icao24List'] = icao24List

        for key in typecodeKeys[15:]:
            i = 0
            icao24List = []
            """
            for model in data['model']:
                # Case: '172' = last 3 characters of 'model'
                if model[-3:] == key[1:]:
                    icao24List.append(data['icao24'][i+1]) # i+1: lines up icao24 w/ typecode
                # # Case: '172' = first 3 characters of 'model'
                elif model[0:3] == key[1:]:
                    icao24List.append(data['icao24'][i+1]) # i+1: lines up icao24 w/ typecode
                i += 1
            # Case: 'C###' typecode
            i = 0
            """
            for typecode in data['typecode']:
                if typecode == key:
                    icao24List.append(data['icao24'][i+1])  # i+1: lines up icao24 w/ typecode
                i += 1
            typecodeDict[key]['icao24List'] = icao24List

        icao24s = open('../../data/flight_plan/icao24s_'
                       + str(self.timestamp) + '.p', 'wb')
        pickle.dump(typecodeDict, icao24s)
        icao24s.close()

    def update_OpenSkyApi(self):

        typecodeDict = pickle.load(open(self.filepath + self.airframe
                                        + str(self.timestamp) + '.p', 'rb'))

        ########################################################################
        # Creating list of timestamps over 24 hour period (separated by 15 minutes)
        timestampList = np.array([])
        for i in range(0, 24*60, 15):
            timestampList = np.append(timestampList, self.timestamp + i*60)

        ########################################################################
        # Attempting to speed up program by inputting a list of icao24s into get_states().
        #   - First, invalid icao24s (len(icao24) != 6) are removed
        #   - Then, the 'filtered' list is inputted into get_states()
        #   - Problem: only three states are returned from get_states()
        #   - Works for A350, A380
        i = 0
        while i < len(typecodeDict[self.typecode]['icao24List']):
            if len(typecodeDict[self.typecode]['icao24List'][i]) != 6:
                typecodeDict[self.typecode]['icao24List'] = np.append(
                    typecodeDict[self.typecode]['icao24List'][:i], typecodeDict[self.typecode]
                    ['icao24List'][i+1:])
            else:
                i += 1

        ########################################################################

        # MAIN PROGRAM USED TO GENERATE PDF AT ONE TIMESTAMP

        # for ica24 in typecodeDict[self.typecode]['icao24List']:
        #     api = OpenSkyApi('jplilly25','Crossfit25')
        #     try:
        #         state = api.get_states(time_secs=self.timestamp,icao24=ica24)
        #         if state.states:
        #             if (state.states[0].velocity != 0) and (state.states[0].vertical_rate != None):
        #                 self.velocity = np.append(self.velocity,
        #                                     np.array(state.states[0].velocity))
        #                 self.vertrate = np.append(self.vertrate,
        #                                     np.array(state.states[0].vertical_rate))
        #     except:
        #         pass# print(ica24)

        ########################################################################
        # Tested to see if there was a limit on number of icao24s in list.
        #   Does not work when inputting full list at once
        #   States found when len(list) == 500
        if len(typecodeDict[self.typecode]['icao24List']) > 500:
            typecodeDict[self.typecode]['icao24List'] = typecodeDict[self.typecode]['icao24List'][500:1000]
        ########################################################################

        # for timestamp in timestampList:
        #     api = OpenSkyApi('jplilly25','Crossfit25')
        #     try:
        #         state = api.get_states(time_secs=timestamp,icao24=typecodeDict[self.typecode]['icao24List'])
        #         try:
        #             for n in range(len(state.states)):
        #                 if state.states[n]:
        #                     if (state.states[n].velocity != 0) and (state.states[n].vertical_rate != None):
        #                         self.velocity = np.append(self.velocity, state.states[n].velocity)
        #                         self.vertrate = np.append(self.vertrate, state.states[n].vertical_rate)
        #         except:
        #             pass
        #     except:
        #         print('get_states() failed to pull data from OpenSky-Network')
        ########################################################################
        # Running until ~250 data points are found.

        timestamp = self.timestamp  # did not want to change self.timestamp value

        while len(self.velocity) < 1000:
            print(len(self.velocity))
            # Checks to see if state present at timestamp
            api = OpenSkyApi('jplilly25', 'Crossfit25')
            state = api.get_states(time_secs=timestamp,
                                   icao24=typecodeDict[self.typecode]['icao24List'])

            # try/except block acts as if statement (i.e. if state found, the code
            #   in the try block will execute)
            try:
                # The previous 10 minutes and next 10 minutes from timestamp are
                #   searched for states (every minute is checked)
                t1 = timestamp - (10*60)
                t2 = timestamp + (10*60)
                timestampList = np.linspace(t1, t2, 21)
                for t in timestampList:
                    api = OpenSkyApi('jplilly25', 'Crossfit25')
                    state = api.get_states(
                        time_secs=t, icao24=typecodeDict[self.typecode]['icao24List'])
                    try:
                        for n in range(len(state.states)):
                            if state.states[n]:
                                if (state.states[n].velocity != 0) and (state.states[n].vertical_rate != None):
                                    self.velocity = np.append(
                                        self.velocity, state.states[n].velocity)
                                    self.vertrate = np.append(
                                        self.vertrate, state.states[n].vertical_rate)
                                if state.states[n].velocity > 80:
                                    print(state.states[n].icao24)
                    except:
                        pass
            except:
                pass

            # timestamp value is updated to the next 15 minute mark
            timestamp += 15*60

        icao24s = open(self.filepath + str(self.typecode) + '_' +
                       str(self.timestamp) + '.p', 'wb')
        pickle.dump({'velocity': self.velocity, 'angleOfAttack': self.angleOfAttack,
                     'vertrate': self.vertrate}, icao24s)
        icao24s.close()

    def calculate_angle_of_attack(self, velocity, weight):

        angleOfAttack = np.array([])

        # Data for Cessna 172 (cruise altitude and cruise speed)
        # W = 1016.047 * 9.81 # [kg*m*s^-2]
        W = weight * 9.81  # [kg*m*s^2]
        rho = 0.770488088  # [kg/m^3]
        # u = 62.57 # [m/s]: cruise for Cessna 172
        S = 16.1651  # [m^2]
        Cl_alpha = 0.0776
        Cl_o = 0.33
        incidenceAngle = 1  # [deg]

        cos_alpha = np.linspace(np.cos(-5/180*np.pi), np.cos(15/180*np.pi), len(velocity))
        cos_alpha = np.mean(cos_alpha)
        constant = 2*W/rho/(velocity)**2/S

        Cl = (1/cos_alpha) * constant

        # Defining Cl as average of possible Cls for Cessna 172 assuming -5 < AoA < 15 [deg]
        # Cl = np.mean(Cl)

        initial_AoA = (Cl-Cl_o)/Cl_alpha + incidenceAngle  # [deg]

        # Implementing fixed-point iteration:
        err = 0.001
        j = 0
        for initial_AoA_value in initial_AoA:
            next_AoA = 0
            i = 0
            while (abs(initial_AoA_value-next_AoA) > err).any() and \
                    i < 10000:
                # if statement included to set the initial_AoA_value to the
                #   previously calculated AoA
                if (next_AoA != np.array([0])).any():
                    initial_AoA_value = next_AoA

                # Calculation of part of the calculation for AoA (depends on
                #   velocity which is an array, therefore, need index)
                constant = 2*W/rho/(velocity[j])**2/S

                # Calculation of coefficient of lift and AoA
                Cl = (1/np.cos(initial_AoA_value/180*np.pi)) * constant
                next_AoA = (Cl-Cl_o)/Cl_alpha + incidenceAngle
                i += 1

            j += 1

            # Adding converged AoA value to list of AoA's for later use
            angleOfAttack = np.append(angleOfAttack, next_AoA)

        # self.angleOfAttack = initial_AoA
        return angleOfAttack

    def generate_pdf(self, parameters):
        """Description"""
        if parameters is None:
            parameters = np.vstack([self.angleOfAttack, self.velocity])

        # Filtering a couple of data points that are outliers
        i = 0
        count = 0  # Included to check how many outliers were filtered out
        while i < len(self.angleOfAttack):
            if self.angleOfAttack[i] < -5 or self.angleOfAttack[i] > 30:
                self.angleOfAttack = np.append(self.angleOfAttack[:i],
                                               self.angleOfAttack[i+1:])
                self.velocity = np.append(self.velocity[:i], self.velocity[i+1:])
                count += 1
            else:
                i += 1

        # Calculate the PDF of velocity vs. angle of attack for a single weight.
        values = np.vstack([self.angleOfAttack, self.velocity])
        pdf = gaussian_kde(values)

        # Find conditional probability of (angle of attack, velocity) given
        #   a certain weight.
        #   Weight of aircraft is assumed to have a uniform distribution.
        #   P(W) = 1/(max_weight - min_weight)
        #   P((alpha, v) and W) = P((alpha, v) | W) * P(W)

        min_weight = 618  # [kg] = 1,363 [lb]
        max_weight = 919  # [kg] = 2027 [lb]

        # weight_distribution = 1/(max_weight-min_weight)
        pdf_points = pdf(parameters)  # *weight_distribution  # Conditional probability calculation

        return pdf_points.reshape(parameters[0])

    def plot_pdf(self, parameters=None,
                 xgrid=np.linspace(-5, 35, 1000),
                 ygrid=np.linspace(20, 75, 1000)):
        """Generate contour plot visualizing PDF of velocity vs. angle of
        attack."""

        pdf_points = self.generate_pdf(parameters)

        X, Y = np.meshgrid(xgrid, ygrid)

        Z = np.reshape(pdf_points.T, X.shape)

        ########################################################################
        # Generate custom contour levels to better visualize data for A380
        #   ('higher definition' at lower levels of probability)

        # levels = np.array([0.0000])#[0.000, 0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.001, 0.007])
        #
        # for i in range(0,16):
        #     levels = np.append(levels, levels[i]+0.000025)
        # #     levels = np.append(levels, 10**(i-10))
        #
        # for i in range(0,8):
        #     levels = np.append(levels, levels[i+16]+0.0004)
        # # levels = np.append(levels, [0.003, 0.005, 0.007])
        # # print(levels)
        ########################################################################

        # Levels are manually set to better visualize distribution
        #   NOTE: not set for contour plot considering conditional probability of
        #       weight and (angle of attack, velocity)

        levels = np.array([0.0000])

        for i in range(8):
            levels = np.append(levels, levels[i]+0.0001/8)

        levels = np.append(levels, [0.000375, 0.0015, 0.003, 0.0045, 0.006, 0.0075, 0.009])

        levels = np.array([0.0000, 0.0005, 0.001, 0.0015, 0.003, 0.0045, 0.006,
                           0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.0275])

        fig = plt.figure()
        plt.contourf(X, Y, Z)  # , levels=levels)
        # plt.title('PDF of Velocity vs. Vertical Velocity for %s' % self.typecode)
        # plt.title('PDF of Velocity vs. Angle of Attack for %s' % self.typecode)
        # plt.xlabel('Vertical Velocity [m/s]')
        plt.xlabel('Approximated Angle of Attack [degrees]')
        plt.ylabel('Velocity [m/s]')
        plt.xlim(-2, 35)
        plt.colorbar()

        plt.savefig('../../data/flight_plan/pdf_contours/' + self.typecode + '/'
                    + self.plot_filename)

    def plot_scatter(self):
        """Description"""

        ########################################################################
        # Generating plot for Cessna 172 with cruise and speed of sound plotted.
        cruise_C172 = 62.59  # [m/s]
        # speed_of_sound = 294.9 # [m/s] at altitude = 43,100 [ft]

        x_speeds = np.linspace(-35, 80, 5000)
        cruising = np.ones(len(x_speeds)) * cruise_C172
        # sounds = np.ones(len(x_speeds)) * speed_of_sound
        ########################################################################

        fig = plt.figure()
        # plt.plot(x_speeds, cruising, 'r', zorder = 2, label='Cruise Speed (Mach 0.85)')
        plt.plot(x_speeds, cruising, 'r', label='Cruise Speed = 140 [mph]')
        # plt.plot(x_speeds, sounds, 'g', zorder = 3, label='Speed of Sound \n(Altitude = 43,100 [ft])')
        plt.scatter(self.angleOfAttack, self.velocity, marker='.')
        plt.xlabel('Approximated Angle of Attack [degrees]')
        plt.ylabel('Velocity [m/s]')
        plt.xlim(0, 35)
        # plt.ylim(0,70)
        plt.legend(loc=0, fontsize=7, facecolor='w', framealpha=1.0)
        plt.grid(True)

        plt.savefig('../../data/flight_plan/pdf_contours/' + self.typecode + '/'
                    + self.scatter_filename)

    def plot_weight(self):
        """Description"""

        def compute_AoAs():
            # self.vertrate = np.array(data['vertrate'], dtype='float')
            # self.angleOfAttack = self.calculate_angle_of_attack(self.velocity, weight=882)

            # np.min(self.velocity),np.max(self.velocity),1000)
            velocity = np.linspace(10, 75, 1000)
            velocity = self.velocity
            empty_weight = 618.61  # 1363.81 [lb]
            pilot_weight = 62.14  # 137 [lb]
            fuel_weight = 114.31  # 252 [lb]
            fuel_percentage = 0.5  # chosen out of thin air

            mean_weight = empty_weight + 1.5*pilot_weight + (fuel_percentage*fuel_weight)
            max_weight = empty_weight + 3*pilot_weight + fuel_weight
            min_weight = empty_weight

            percentages = [0.05]
            percent_weight_change = [p*mean_weight for p in percentages]

            pwp = [pw+mean_weight for pw in percent_weight_change]
            pwn = [mean_weight-pw for pw in percent_weight_change]
            pwn.reverse()

            weight = [min_weight, pwn[0], mean_weight, pwp[0], max_weight]

            AoA = {str(w): [] for w in weight}

            for w in weight:
                AoA['%s' % str(w)] = self.calculate_angle_of_attack(velocity, weight=w)

            AoAs = open('../../data/flight_plan/pdf_contours/C172/AoAs_5%.p',
                        'wb')
            pickle.dump({'weight': weight, 'AoA': AoA, 'velocity': velocity},
                        AoAs)
            AoAs.close()

        def gnome_sort(array):

            pos = 0
            swapCount = 0

            rows = array.shape[0]
            columns = array.shape[1]

            array_dict = {array[0, i]: [array[0, i], array[1, i]] for i in range(columns)}
            array = array[0, :]

            # The loop will run until the position checked is the last position in the
            #   array.
            # Length of rows in array is used below to terminate after top row is sorted
            while pos < columns:
                # if the value at a position is greater than the one to the left of it,
                # the position is moved one space to the right
                if (pos == 0) or (array[pos] >= array[pos-1]).any():
                    pos += 1
                else:
                    # the value at a position is flipped with the value at the position
                    # to the left of it and the position is moved one space to the left
                    placeholder = array[pos]
                    array[pos] = array[pos-1]
                    array[pos-1] = placeholder
                    swapCount += 1
                    pos -= 1

            # Rebuild array from array_dict and sorted 1-D array
            new_array = np.zeros((2, len(array)))
            for j in range(len(array)):
                for i in range(2):
                    new_array[i, j] = array_dict[array[j]][i]

            return new_array

        # compute_AoAs()

        data = pickle.load(open('../../data/flight_plan/pdf_contours/C172/AoAs_5%.p', 'rb'))

        labels = ['Min Weight = 1,363 [lb]', '-5% Mean Weight = 1,546 [lb]',
                  'Mean Weight = 1,627 [lb]', '+5% Mean Weight = 1,708 [lb]', 'Max Weight = 2027 [lb]']

        fig = plt.figure()

        # Taken out so that the data ranges plotted could be changed individually
        # for i in range(len(data['weight'])):
        #     v_aoa = np.stack((data['AoA'][str(data['weight'][i])], data['velocity']))
        #     v_aoa_1 = gnome_sort(v_aoa)
        #     plt.plot(v_aoa_1[0,20:],v_aoa_1[1,20:])

        v_aoa = np.stack((data['AoA'][str(data['weight'][0])], data['velocity']))
        v_aoa_1 = gnome_sort(v_aoa)
        v_aoa = np.stack((data['AoA'][str(data['weight'][1])], data['velocity']))
        v_aoa_2 = gnome_sort(v_aoa)
        v_aoa = np.stack((data['AoA'][str(data['weight'][2])], data['velocity']))
        v_aoa_3 = gnome_sort(v_aoa)
        v_aoa = np.stack((data['AoA'][str(data['weight'][3])], data['velocity']))
        v_aoa_4 = gnome_sort(v_aoa)
        v_aoa = np.stack((data['AoA'][str(data['weight'][4])], data['velocity']))
        v_aoa_5 = gnome_sort(v_aoa)

        cruise_C172 = 62.59  # [m/s]
        x_speeds = np.linspace(-35, 80, 5000)
        cruising = np.ones(len(x_speeds)) * cruise_C172

        plt.plot(v_aoa_1[0, 20:], v_aoa_1[1, 20:], label=labels[0])
        plt.plot(v_aoa_2[0, 20:], v_aoa_2[1, 20:], label=labels[1])
        plt.plot(v_aoa_3[0, 10:], v_aoa_3[1, 10:], label=labels[2])
        plt.plot(v_aoa_4[0, 20:], v_aoa_4[1, 20:], label=labels[3])
        plt.plot(v_aoa_5[0, 30:], v_aoa_5[1, 30:], label=labels[4])
        plt.plot(x_speeds, cruising, '--', label='Cruise Speed = 140 [mph]')
        # for w in data['weight']:
        # plt.scatter(data['AoA']['%s' % str(w)], data['velocity'], label=labels)
        plt.xlabel('Approximated Angle of Attack [degrees]')
        plt.ylabel('Velocity [m/s]')
        plt.xlim(-1, 30)
        plt.ylim(0, 75)
        plt.legend(fontsize='small', loc='upper right', framealpha=1.0)
        plt.grid(True)

        plt.savefig('../../data/flight_plan/pdf_contours/C172/weight_pertubations_5%.png')

    def retrieve_data(self):
        data = pickle.load(open(self.filepath + str(self.typecode) + '_' +
                                str(self.timestamp) + '.p', 'rb'))

        self.velocity = np.array(data['velocity'], dtype='float')
        self.vertrate = np.array(data['vertrate'], dtype='float')
        self.angleOfAttack = self.calculate_angle_of_attack(self.velocity,
                                                            weight=882)
