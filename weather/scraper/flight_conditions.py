import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fixed_point
from sklearn.neighbors.kde import KernelDensity

from opensky_api import OpenSkyApi
from weather.filehandling import output_reader


class properties(object):
    def __init__(self, inputs):
        for key in inputs:
            setattr(self, key, inputs[key])

        self.weight_min = self.mass_min * 9.81
        self.weight_max = self.mass_max * 9.81


class Airframe(object):
    """ Object to store data from OpenSky and estimate probabilities based on
        the retrieved data

        inputs:
         - airframe: typecode for airframe (e.g. B737)
         - timestamp: Time when OpenSkyApi pulls data from
         - filepath: path to where information is retrieved
         - properties: object properties with all airframe and other relevant
                      properties
       """

    def __init__(self, timestamp=1549729462, airframe='C172',
                 filepath='../../data/flight_plan/v_aoa_pickles/icao24s_',
                 properties=properties):
        self.airframe = airframe
        self.timestamp = timestamp
        self.filepath = filepath
        self.properties = properties

    def update_icao24s(self,
                       filepath='../../data/flight_plan/aircraftDatabases/',
                       filename='aircraftDatabase_1549729462.csv'):
        """Update list of icao24s for all airframes.
        Generates icao24s_self.timestamp.p pickle
        """

        type_structure = ['string' for n in range(3)]

        data = output_reader(filepath + filename, separator=[','],
                             type_structure=type_structure)

        if list(data.keys()) != ['icao24','typecode','model']:
            raise SyntaxError('Workbook must have columns: icao24, typecode,\
                              model')

        airframeDict = {'B737': {}, 'B747': {}, 'B757': {}, 'B767': {},
                        'B777': {}, 'B787': {},'A310': {}, 'A318': {},
                        'A319': {}, 'A320': {}, 'A321': {},'A330': {},
                        'A340': {}, 'A350': {}, 'A380': {}, 'C172': {},
                        'C180': {},'C182': {}}
        airframeKeys = list(airframeDict.keys())
        airframeDict = {key: {'icao24List': []} for key in airframeDict}

        # Pull icao24s for Airbus and Boeing airframes.
        for key in airframeKeys[:15]:
            airframeDict[key]['icao24List'] = [icao24 for airframe in data['typecode'] if airframe[0:3] == key[0:3]]

        # Pull icao24s for Cessna airframes.
        for key in airframeKeys[15:]:
            airframeDict[key]['icao24List'] = [icao24 for airframe in data['model'] if airframe == key]

        icao24s = open('../../data/flight_plan/icao24_lists/icao24s_'
                       + str(self.timestamp) + '.p', 'wb')
        pickle.dump(airframeDict, icao24s)
        icao24s.close()

    def update_OpenSkyApi(self):

        airframeDict = pickle.load(open(self.filepath + self.airframe
                                        + str(self.timestamp) + '.p', 'rb'))

        # Filtering out icao24s that are invalid
        i = 0
        while i < len(airframeDict[self.airframe]['icao24List']):
            if len(airframeDict[self.airframe]['icao24List'][i]) != 6:
                airframeDict[self.airframe]['icao24List'] = np.append(
                    airframeDict[self.airframe]['icao24List'][:i],
                    airframeDict[self.airframe]
                    ['icao24List'][i+1:])
            else:
                i += 1

        # OpenSkyApi only works for lists with lengths less than ~500
        if len(airframeDict[self.airframe]['icao24List']) > 500:
            airframeDict[self.airframe]['icao24List'] = \
            airframeDict[self.airframe]['icao24List'][500:1000]

        timestamp = self.timestamp  # did not want to change self.timestamp value

        while len(self.velocity) < 1000:
            # Checks to see if state present at timestamp
            api = OpenSkyApi('jplilly25', 'Crossfit25')
            state = api.get_states(time_secs=timestamp,
                                   icao24=typecodeDict[self.typecode]['icao24List'])
            try:
                t1 = timestamp - (10*60)
                t2 = timestamp + (10*60)
                timestampList = np.linspace(t1, t2, 21)
                for t in timestampList:
                    api = OpenSkyApi('jplilly25', 'Crossfit25')
                    state = api.get_states(
                        time_secs=t, icao24=airframeDict[self.airframe]
                        ['icao24List'])
                    try:
                        for n in range(len(state.states)):
                            if state.states[n]:
                                if (state.states[n].velocity != 0) and
                                   (state.states[n].vertical_rate != None):
                                    self.velocity = np.append(
                                        self.velocity, state.states[n].velocity)
                                    self.climb_rate = np.append(
                                        self.climb_rate,
                                        state.states[n].vertical_rate)
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
        pickle.dump({'velocity': self.velocity, 'climb_rate': self.climb_rate}, icao24s)
        icao24s.close()

    def calculate_aoa(self, weight, velocity):
        # If input is not list/array convert it to one
        try:
            len(velocity)  # dummy test
        except(TypeError):
            velocity = [velocity]
        try:
            len(weight)  # dummy test
        except(TypeError):
            weight = [weight]

        # Implementing fixed-point iteration
        aoa = []
        for i in range(len(velocity)):
            aoa_i = fixed_point(self._aoa, 2., args=(weight[i], velocity[i]))
            aoa.append(float(aoa_i))
        return(aoa)

    def _aoa(self, aoa, weight, velocity):
        p = self.properties
        cos = np.cos(np.radians(aoa + p.incidence))
        aoa = (2*weight/(p.density*velocity**2*p.planform)*cos -
               p.Cl_0)/p.Cl_alpha
        return np.degrees(aoa)

    def plot_pdf(self, parameters=None,
                 xgrid=np.linspace(-5, 35, 1000),
                 ygrid=np.linspace(20, 75, 1000)):
        """Generate contour plot visualizing PDF of velocity vs. angle of
        attack."""

        X, Y = np.meshgrid(xgrid, ygrid)
        Z = np.exp(self.pdf.score_samples(np.array([X.ravel(), Y.ravel()]).T))
        Z = np.reshape(Z, X.shape)

        plt.figure()
        plt.contourf(X, Y, Z)
        plt.xlabel('Approximated Angle of Attack [degrees]')
        plt.ylabel('Velocity [m/s]')
        # plt.xlim(-2, 35)
        plt.colorbar()
        plt.show()

    def retrieve_data(self):
        data = pickle.load(open(self.filepath + str(self.airframe) + '_' +
                                str(self.timestamp) + '.p', 'rb'))

        self.velocity = np.array(data['velocity'], dtype='float')
        self.velocity = self.velocity.reshape(len(self.velocity), 1)
        self.climb_rate = np.array(data['climb_rate'], dtype='float')
        self.climb_rate = self.climb_rate.reshape(len(self.climb_rate), 1)
        self.pdf_velocity = KernelDensity(kernel='gaussian').fit(self.velocity)
        self.pdf_climb = KernelDensity(kernel='gaussian').fit(self.climb_rate)

    def train_pdf(self, population=100000):
        """Description"""
        p = self.properties
        self.database = []
        for i in range(population):
            weight = np.random.uniform(p.weight_min, p.weight_max)
            velocity = self.pdf_velocity.sample()[0]
            aoa = self.calculate_aoa(weight, velocity)
            self.database.append([aoa[0], velocity[0]])

        self.database = np.array(self.database)
        self.pdf = KernelDensity(kernel='gaussian').fit(self.database)
