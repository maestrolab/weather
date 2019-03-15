"""
Script to generate PDF of velocity vs. approximated angle of attack for
specified classes of aircraft.
"""

import pickle
from opensky_api.python.opensky_api import OpenSkyApi
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from weather.filehandling import output_reader

################################################################################
class Airframe(object):
    """as"""

    def __init__(self, timestamp=1549729462, typecode='B737'):
        self.typecode = typecode #--> (in methods, only pull [0:3])
        self.timestamp = timestamp # Included to tell what csv to use and when to pull OpenSkyApi data from
        self.velocity = np.array([])
        self.vertrate = np.array([])
        self.angleOfAttack = np.array([])
        self.csv_filename = 'aircraftDatabase_1549729462_test.csv' #+ str(self.timestamp) + '_test.csv'
        self.plot_filename = 'converged_v_AoA_500_1000_' + self.typecode + '_24_hours_5_minutes.png'
        self.scatter_filename = 'converged_v_AoA_scatter_500_1000_' + self.typecode + '_24_hours_5_minutes.png'

    def update_csv(self):
        """
        Instructions how to format aircraftDatabase.csv downloaded from
           OpenSky-Network.
        """
        pass # Will include description of how to format csv for proper use

    def read_csv(self, filename):
        """Read aircraft database csv."""

        type_structure = []
        for n in range(3):
            type_structure.append('string')

        data = output_reader('../data/flight_plan/aircraftDatabases/' + filename, separator=[','],
                             type_structure=type_structure)
        return data

    def update_icao24s(self):
        """Update list of icao24s for all typecodes.
        Generates icao24s_self.timestamp.p pickle (all icao24s sorted by typecode)
        """

        data = self.read_csv(self.csv_filename) # Made separate function in case a different filename
                          # is wanted to be used. (type(data) == dictionary)

        typecodeDict = {'B737':{},'B747':{},'B757':{},'B767':{},'B777':{},'B787':{},
                        'A310':{},'A318':{},'A319':{},'A320':{},'A321':{},
                        'A330':{},'A340':{},'A350':{},'A380':{},'C172':{},'C180':{},
                        'C182':{}}
        typecodeKeys = list(typecodeDict.keys())
        typecodeDict = {key:{'icao24List':[], 'manufacturer':''} for key in typecodeDict}

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
            i = 0 # here to offset icao24 row pull
            icao24List = []
            for typecode in data['typecode']:
                if typecode[0:3] == key[0:3]:
                    icao24List.append(data['icao24'][i+1]) # i+1: lines up icao24 w/ typecode
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
                    icao24List.append(data['icao24'][i+1]) # i+1: lines up icao24 w/ typecode
                i += 1
            typecodeDict[key]['icao24List'] = icao24List

        icao24s = open('../data/flight_plan/icao24s_'
                        + str(self.timestamp) + '.p','wb')
        pickle.dump(typecodeDict,icao24s)
        icao24s.close()

    def update_OpenSkyApi(self):

        typecodeDict = pickle.load(open('../data/flight_plan/icao24s_'
                        + str(self.timestamp) + '.p','rb'))

        ########################################################################
        # Creating list of timestamps over 24 hour period (separated by 15 minutes)
        timestampList = np.array([])
        for i in range(0,24*60,5):
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
                typecodeDict[self.typecode]['icao24List'] = np.append(\
                typecodeDict[self.typecode]['icao24List'][:i],typecodeDict[self.typecode]\
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

        for timestamp in timestampList:
            api = OpenSkyApi('jplilly25','Crossfit25')
            try:
                state = api.get_states(time_secs=timestamp,icao24=typecodeDict[self.typecode]['icao24List'])
                try:
                    for n in range(len(state.states)):
                        if state.states[n]:
                            if (state.states[n].velocity != 0) and (state.states[n].vertical_rate != None):
                                self.velocity = np.append(self.velocity, state.states[n].velocity)
                                self.vertrate = np.append(self.vertrate, state.states[n].vertical_rate)
                except:
                    pass
            except:
                print('get_states() failed to pull data from OpenSky-Network')
        ########################################################################

        icao24s = open('../data/flight_plan/v_aoa_pickles/icao24s_' +
                        str(self.typecode) + '_' + str(self.timestamp) + '.p','wb')
        pickle.dump({'velocity':self.velocity,'angleOfAttack':self.angleOfAttack,
                     'vertrate':self.vertrate},icao24s)
        icao24s.close()

    def calculate_angle_of_attack(self, velocity):

        # Data for Cessna 172 (cruise altitude and cruise speed)
        W = 1016.047 * 9.81 # [kg*m*s^-2]
        rho = 0.770488088 # [kg/m^3]
        # u = 62.57 # [m/s]: cruise for Cessna 172
        S = 16.1651 # [m^2]
        Cl_alpha = 0.0776
        Cl_o = 0.33
        incidenceAngle = 1 # [deg]

        cos_alpha = np.linspace(np.cos(-5/180*np.pi), np.cos(15/180*np.pi), len(velocity))
        cos_alpha = np.mean(cos_alpha)
        constant = 2*W/rho/(velocity)**2/S

        Cl = (1/cos_alpha) * constant

        # Defining Cl as average of possible Cls for Cessna 172 assuming -5 < AoA < 15 [deg]
        # Cl = np.mean(Cl)

        initial_AoA = (Cl-Cl_o)/Cl_alpha + incidenceAngle # [deg]

        # Implementing fixed-point iteration:
        err = 0.001
        j = 0
        for initial_AoA_value in initial_AoA:
            next_AoA = 0
            i = 0
            while (abs(initial_AoA_value-next_AoA) > err).any() and \
                    i < 10000000:
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
            self.angleOfAttack = np.append(self.angleOfAttack, next_AoA)

        # self.angleOfAttack = initial_AoA
        return self.angleOfAttack

    def generate_pdf(self):
        """Description"""

        data = pickle.load(open('../data/flight_plan/v_aoa_pickles/icao24s_' +
                        str(self.typecode) + '_' + str(self.timestamp) + '.p','rb'))

        self.velocity = np.array(data['velocity'], dtype='float')
        self.vertrate = np.array(data['vertrate'], dtype='float')
        self.angleOfAttack = self.calculate_angle_of_attack(self.velocity)

        values = np.vstack([self.angleOfAttack, self.velocity])
        # values = np.vstack([self.angleOfAttack, self.velocity])
        pdf = gaussian_kde(values)

        return pdf

    def plot_pdf(self):
        """Generate contour plot visualizing PDF of velocity vs. angle of attack."""

        pdf = self.generate_pdf()

        xgrid = np.linspace(-4.5,30,1000)# np.linspace(np.amin(self.angleOfAttack), np.amax(self.angleOfAttack), 1000)
        ygrid = np.linspace(20,75,1000)# np.linspace(np.amin(self.velocity), np.amax(self.velocity), 1000)

        X, Y = np.meshgrid(xgrid,ygrid)
        positions = np.vstack([X.ravel(), Y.ravel()])

        Z = np.reshape(pdf(positions).T, X.shape)

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

        levels = np.array([0.0000])

        for i in range(8):
            levels = np.append(levels, levels[i]+0.0001/8)

        levels = np.append(levels, [0.000375, 0.0015, 0.003, 0.0045, 0.006, 0.0075, 0.009])

        fig = plt.figure()
        plt.contourf(X,Y,Z, levels=levels)
        # plt.title('PDF of Velocity vs. Vertical Velocity for %s' % self.typecode)
        # plt.title('PDF of Velocity vs. Angle of Attack for %s' % self.typecode)
        # plt.xlabel('Vertical Velocity [m/s]')
        plt.xlabel('Approximated Angle of Attack [degrees]')
        plt.ylabel('Velocity [m/s]')
        #plt.xlim(-4.5,35)
        #plt.ylim(0,70)
        plt.colorbar()

        plt.savefig('../data/flight_plan/pdf_contours/' + self.typecode + '/'
                    + self.plot_filename)
        # plt.show()

    def plot_scatter(self):
        """Description"""

        data = pickle.load(open('../data/flight_plan/v_aoa_pickles/icao24s_' +
                        str(self.typecode) + '_' + str(self.timestamp) + '.p','rb'))

        self.velocity = np.array(data['velocity'], dtype='float')
        self.vertrate = np.array(data['vertrate'], dtype='float')
        self.angleOfAttack = self.calculate_angle_of_attack(self.velocity)

        ########################################################################
        # Generating plot for A380 with cruise and speed of sound plotted.
        # cruise_A380 = 250.83 # [m/s]
        # speed_of_sound = 294.9 # [m/s] at altitude = 43,100 [ft]
        #
        # x_speeds = np.linspace(-30,30,5000)
        # cruising = np.ones(len(x_speeds)) * cruise_A380
        # sounds = np.ones(len(x_speeds)) * speed_of_sound
        ########################################################################

        ########################################################################
        # Generating plot for Cessna 172 with cruise and speed of sound plotted.
        cruise_C172 = 62.59 # [m/s]
        # speed_of_sound = 294.9 # [m/s] at altitude = 43,100 [ft]

        x_speeds = np.linspace(-35,80,5000)
        cruising = np.ones(len(x_speeds)) * cruise_C172
        # sounds = np.ones(len(x_speeds)) * speed_of_sound
        ########################################################################

        fig = plt.figure()
        # plt.plot(x_speeds, cruising, 'r', zorder = 2, label='Cruise Speed (Mach 0.85)')
        plt.plot(x_speeds, cruising, 'r', zorder = 2, label='Cruise Speed = 140 [mph]')
        # plt.plot(x_speeds, sounds, 'g', zorder = 3, label='Speed of Sound \n(Altitude = 43,100 [ft])')
        plt.scatter(self.angleOfAttack, self.velocity, zorder = 1)
        plt.xlabel('Approximated Angle of Attack [degrees]')
        # plt.xlabel('Angle of Attack [degrees]')
        plt.ylabel('Velocity [m/s]')
        plt.xlim(0,30)
        # plt.ylim(0,70)
        plt.legend(loc=0, fontsize=7, facecolor='w', framealpha=1.0)
        plt.grid(True)

        plt.savefig('../data/flight_plan/pdf_contours/' + self.typecode + '/'
                    + self.scatter_filename)
        # plt.show()

################################################################################

if __name__ == '__main__':
    typecodeList = ['B737','B747','B757','B767','B777','B787',
                    'A310','A318','A319','A320','A321',
                    'A330','A340','A350','A380','C172','C180',
                    'C182']

    airFrame = Airframe(typecode=typecodeList[15], timestamp=1549036800)
    # airFrame.update_icao24s()
    # airFrame.update_OpenSkyApi() # must run for each typecode before pdf generation
    # pdf = airFrame.generate_pdf()
    # airFrame.plot_pdf()
    # airFrame.plot_scatter()
