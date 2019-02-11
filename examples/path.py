"""
Script to generate PDF of velocity vs. angle of attack for specified classes of
aircraft.
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

    def __init__(self, timestamp, typecode='B737'):
        self.typecode = typecode #--> (in methods, only pull [0:3])
        self.timestamp = timestamp # Included to tell what csv to use and when to pull OpenSkyApi data from
        self.velocity = np.array([])
        self.vertrate = np.array([])
        self.angleOfAttack = np.array([])
        self.csv_filename = 'aircraftDatabase_' + str(self.timestamp) + '_test.csv'
        self.plot_filename = 'v_aoa_' + self.typecode + '_' + str(self.timestamp) + '.png'

    def update_csv(self):
        """
        Instructions how to format aircraftDatabase.csv downloaded from
           OpenSky-Network.
        """
        pass # Will include description of how to format csv for proper use

    def read_csv(self, filename):
        """Read aircraft database csv."""

        type_structure = []
        for n in range(2):
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
                        'A330':{},'A340':{},'A350':{},'A380':{}}
        typecodeKeys = list(typecodeDict.keys())
        typecodeDict = {key:{'icao24List':[]} for key in typecodeDict}

        for key in typecodeKeys:
            i = 0
            icao24List = []
            for typecode in data['typecode']:
                if typecode[0:3] == key[0:3]:
                    icao24List.append(data['icao24'][i+1]) # i+1: lines up icao24 w/ typecode
                i += 1
            typecodeDict[key]['icao24List'] = icao24List

        icao24s = open('../data/flight_plan/icao24s_'
                        + str(self.timestamp) + '.p','wb')
        pickle.dump(typecodeDict,icao24s)
        icao24s.close()

    def update_OpenSkyApi(self):
        """Description --> updates values for velocity and angle of attack.
        Generates v_aoa_pickle for self.typecode ('B737','B747',etc.)
        """

        typecodeDict = pickle.load(open('../data/flight_plan/icao24s_'
                        + str(self.timestamp) + '.p','rb'))

        for ica24 in typecodeDict[self.typecode]['icao24List']:
            api = OpenSkyApi('jplilly25','Crossfit25')
            try:
                state = api.get_states(time_secs=self.timestamp,icao24=ica24)
                if state.states:
                    if (state.states[0].velocity != 0) and (state.states[0].vertical_rate != None):
                        self.velocity = np.append(self.velocity,
                                            np.array(state.states[0].velocity))
                        self.vertrate = np.append(self.vertrate,
                                            np.array(state.states[0].vertical_rate))
            except:
                print(ica24)

        ################################################################################
        # Attempting to speed up program by inputting a list of icao24s into get_states().
        #   - First, invalid icao24s (len(icao24) != 6) are removed
        #   - Then, the 'filtered' list is inputted into get_states()
        #   - Problem: only three states are returned from get_states()
        #
        # i = 0
        # while i < len(typecodeDict[self.typecode]['icao24List']):
        #     if len(typecodeDict[self.typecode]['icao24List'][i]) != 6:
        #         typecodeDict[self.typecode]['icao24List'] = np.append(typecodeDict[self.typecode]['icao24List'][:i],typecodeDict[self.typecode]['icao24List'][i+1:])
        #     else:
        #         i += 1
        #
        # api = OpenSkyApi('jplilly25','Crossfit25')
        # state = api.get_states(time_secs=self.timestamp,icao24=typecodeDict[self.typecode]['icao24List'])
        # for n in range(len(state.states)):
        #     if state.states:
        #         self.velocity = np.append(self.velocity, state.states[n].velocity)
        #         self.vertrate = np.append(self.vertrate, state.states[n].vertical_rate)
        ################################################################################

        ################################################################################
        # Removing all instances where vertical_rate == None (assuming onground)
        #
        # i = 0
        # while i < len(self.vertrate):
        #     if self.vertrate[i] == None:
        #         self.vertrate = np.append(self.vertrate[:i], self.vertrate[i+1:])
        #         self.velocity = np.append(self.velocity[:i], self.velocity[i+1:])
        #     else:
        #         i += 1
        ################################################################################

        for n in range(len(self.velocity)):
            self.angleOfAttack = np.append(self.angleOfAttack,
                                    np.arctan(self.vertrate[n]/self.velocity[n]))

        self.angleOfAttack = self.angleOfAttack*(180/np.pi) # rad to degrees
        incidenceAngle = 6 # degrees

        self.angleOfAttack += incidenceAngle


        icao24s = open('../data/flight_plan/v_aoa_pickles/icao24s_' +
                        str(self.typecode) + '_' + str(self.timestamp) + '.p','wb')
        pickle.dump({'velocity':self.velocity,'angleOfAttack':self.angleOfAttack},icao24s)
        icao24s.close()

    def generate_pdf(self):
        """Description"""

        data = pickle.load(open('../data/flight_plan/v_aoa_pickles/icao24s_' +
                        str(self.typecode) + '_' + str(self.timestamp) + '.p','rb'))

        self.velocity = np.array(data['velocity'], dtype='float')
        self.angleOfAttack = np.array(data['angleOfAttack'], dtype='float')

        values = np.vstack([self.angleOfAttack, self.velocity])
        pdf = gaussian_kde(values)

        return pdf

    def plot_pdf(self):
        """Generate contour plot visualizing PDF of velocity vs. angle of attack."""

        pdf = self.generate_pdf()

        xgrid = np.linspace(0, 11, 1000)# np.amin(self.angleOfAttack), np.amax(self.angleOfAttack), 1000)
        ygrid = np.linspace(0, 350, 1000)# np.amin(self.velocity), np.amax(self.velocity), 1000)

        X, Y = np.meshgrid(xgrid,ygrid)
        positions = np.vstack([X.ravel(), Y.ravel()])

        Z = np.reshape(pdf(positions).T, X.shape)

        fig = plt.figure()
        plt.contourf(X,Y,Z)
        plt.title('PDF of Velocity vs. Angle of Attack for %s' % self.typecode)
        plt.xlabel('Angle of Attack [degrees]')
        plt.ylabel('Velocity [m/s]')
        plt.colorbar()

        plt.savefig('../data/flight_plan/pdf_contours/' + self.plot_filename)
        plt.show()
################################################################################

if __name__ == '__main__':
    typecodeList = ['B737','B747','B757','B767','B777','B787',
                    'A310','A318','A319','A320','A321',
                    'A330','A340','A350','A380']

    airFrame = Airframe(timestamp=1549729462,typecode=typecodeList[5])
    # airFrame.update_icao24s()
    airFrame.update_OpenSkyApi() # must run for each typecode before pdf generation
    # airFrame.generate_pdf()
    # airFrame.plot_pdf()
