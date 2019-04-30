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
         - csv_filepath: path to csv file containing airframe icao24s
         - properties: object properties with all airframe and other relevant
                      properties
       """

    def __init__(self, timestamp=1549729462, airframe='C172',
                 filepath='../../data/flight_plan/v_aoa_pickles/icao24s_',
                 csv_filepath='../../data/flight_plan/aircraftDatabases/' +
                 'aircraftDatabase.csv', properties=properties):
        self.airframe = airframe
        self.timestamp = timestamp
        self.filepath = filepath
        self.csv_filepath = csv_filepath
        self.properties = properties

    def _update_icao24s(self, csv_filepath, desired_airframes):
        """Update list of icao24s for all airframes."""

        type_structure = ['string' for n in range(3)]

        data = output_reader(csv_filepath, separator=[','],
                             type_structure=type_structure)

        if list(data.keys()) != ['icao24','typecode','model']:
            raise SyntaxError('Workbook must have columns: icao24, typecode, '
                              + 'model')

        airframeDict = {airframe:{'icao24List':[]} for airframe in
                        desired_airframes}

        for key in desired_airframes:
            # Pull icao24s for Cessna airframes.
            if key[0] == 'C':
                airframeDict[key]['icao24List'] = [data['icao24'][i] for i in
                    range(len(data['model'])) if data['model'][i][0:3] == key[1:]]
            # Pull icao24s for Airbus, Boeing, and all other airframes.
            else:
                airframeDict[key]['icao24List'] = [data['icao24'][i+1] for i in
                 range(len(data['typecode'])) if data['typecode'][i][0:3] ==
                 key[0:3]]

        return airframeDict

    def update_OpenSkyApi(self, desired_airframes=['C172'],
                          num_data_points=1000, time_increment=15,
                          save_data=True):

        airframeDict = self._update_icao24s(self.csv_filepath, desired_airframes)

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
            airframeDict[self.airframe]['icao24List'][:500]

        timestamp = self.timestamp
        velocity = []
        climb_rate = []
        altitude = []

        while len(velocity) < num_data_points:
            # Checks to see if airframe data is present at the current timestamp
            api = OpenSkyApi('jplilly25', 'Crossfit25')
            state = api.get_states(time_secs=timestamp,
                                   icao24=airframeDict[self.airframe]['icao24List'])

            # If a airframe data is present at the current timestamp, sample
            #   surrounding timestamps for more airframe data.
            if state.states:
                t1 = timestamp - ((time_increment-1)*60)
                t2 = timestamp + ((time_increment-1)*60)
                timestampList = np.linspace(t1, t2, (t2-t1)/60+1)
                for t in timestampList:
                    api = OpenSkyApi('jplilly25', 'Crossfit25')
                    state = api.get_states(
                        time_secs=t, icao24=airframeDict[self.airframe]
                        ['icao24List'])
                    for n in range(len(state.states)):
                        if state.states[n]:
                            if (state.states[n].velocity != 0) and\
                               (state.states[n].vertical_rate != None) and\
                               (state.states[n].velocity != None):
                                velocity = np.append(velocity,
                                            state.states[n].velocity)
                                climb_rate = np.append(climb_rate,
                                    state.states[n].vertical_rate)
                                altitude = np.append(altitude,
                                    state.states[n].baro_altitude)
            timestamp += time_increment*60

        # Storing values
        if save_data:
            icao24s = open(self.filepath + str(self.airframe) + '_' +
                           str(self.timestamp) + '.p', 'wb')
            pickle.dump({'velocity': velocity, 'climb_rate': climb_rate,
                         'altitude': altitude}, icao24s)
            icao24s.close()
        self.velocity = np.array(velocity, dtype='float')
        self.climb_rate = np.array(climb_rate, dtype='float')
        self.altitude = np.array(altitude, dtype='float')

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
        plt.colorbar()
        plt.show()

    def retrieve_data(self, load_data=False):
        if load_data:
            data = pickle.load(open(self.filepath + str(self.airframe) + '_' +
                                    str(self.timestamp) + '.p', 'rb'))

            self.velocity = np.array(data['velocity'], dtype='float')
            self.climb_rate = np.array(data['climb_rate'], dtype='float')
        self.velocity = self.velocity.reshape(len(self.velocity), 1)
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
