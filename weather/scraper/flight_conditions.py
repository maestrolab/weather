import time
import pickle
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from scipy.optimize import fixed_point
from sklearn.neighbors.kde import KernelDensity
from pandas import read_csv

from opensky_api import OpenSkyApi


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

        compatible airframes: (below are just a few examples)
         - Airbus aircraft: "A320" and other Airbus planes in a similar manner
         - Boeing aircraft: "B737" and other Boeing planes in a similar manner
         - Cessna aircraft: "C172", "C180", and "C182" are the main ones
         - King Air aircraft: "King Air"
         - Pegasus aircraft: "Pegasus"
         - Piper aircraft: "Piper"
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
        '''Update list of icao24s for all airframes.'''

        # Read csv file
        dtype = {'icao24':'str', 'typecode':'str', 'model':'str'}
        data = read_csv(self.csv_filepath, sep=',', header=0,
                        encoding='latin-1', skiprows=[1], dtype=dtype)

        # Convert Dataframe to dictionary; convert 'model' and 'typecode' values
        #   to strings
        data = {'icao24':data['icao24'].values,
                'typecode':data['typecode'].values,
                'model':data['model'].values}
        data['model'] = [str(data['model'][i]) for i in
                         range(len(data['model']))]
        data['typecode'] = [str(data['typecode'][i]) for i in
                         range(len(data['typecode']))]

        if list(data.keys()) != ['icao24','typecode','model']:
            raise SyntaxError('Workbook must have columns: icao24, typecode, '
                              + 'model')

        airframeDict = {airframe:{'icao24List':[]} for airframe in
                        desired_airframes}

        # To add compatible airframes, copy and paste the code under the 'Piper'
        #   elif and change the conditional statement to identify the desired aircraft

        for key in desired_airframes:
            # Pull icao24s for Cessna airframes.
            if key[0] == 'C':
                airframeDict[key]['icao24List'] = [data['icao24'][i] for i in
                 range(len(data['model'])) if data['model'][i][0:3] == key[1:]]
            # icao24s for general aircraft (excluding Airbus and Boeing)
            elif key[0:2] != 'B7' and key[0:2] != 'A3':
                airframeDict[key]['icao24List'] = [data['icao24'][i] for i in
                range(len(data['model'])) if key.lower() in data['model'][i].lower()]
            # Pull icao24s for Airbus and Boeing
            else:
                airframeDict[key]['icao24List'] = [data['icao24'][i] for i in
                 range(len(data['typecode'])) if data['typecode'][i][0:3] ==
                 key[0:3]]

        return airframeDict

    def update_OpenSkyApi(self, desired_airframes=['C172'],
                          num_data_points=1000, time_increment=15,
                          save_data=True, api_username='jplilly25',
                          api_password='flightparameters'):

        def filter_icao24s(icao24_list):
            '''filter_icao24s removes icao24 codes that are invalid (length is
            less than 6 characters)'''
            i = 0
            while i < len(icao24_list):
                if len(icao24_list[i]) != 6:
                    icao24_list = np.append(icao24_list[:i],icao24_list[i+1:])
                else:
                    i += 1

                return icao24_list

        def configure_icao24s(icao24_list):
            '''configure_icao24s converts a list of icao24s to a dictionary of
            icao24 lists each with a length equal to or less than 500 elements'''
            icao24_dictionary = {}
            if len(icao24_list) > 500:
                num_lists = np.ceil(len(icao24_list)/500)
                for n in range(int(num_lists)):
                    try:
                        icao24_dictionary[n] = icao24_list[0+500*n:500+500*n]
                    except(IndexError):
                        icao24_dictionary[n] = icao24_list[0+500*n:]
            else:
                icao24_dictionary[0] = icao24_list

            return icao24_dictionary

        def scrape_opensky_data(timestamp_list, icao24s, flight_parameters):
            '''scrape_opensky_data scrapes data at a given timestamp for a list
            of icao24s.'''
            for t in timestamp_list:
                api = OpenSkyApi(api_username, api_password)
                state = api.get_states(time_secs=t, icao24=icao24s)
                if state is not None:
                    for n in range(len(state.states)):
                        if state.states[n]:
                            if (state.states[n].velocity != 0) and\
                               (state.states[n].vertical_rate != None) and\
                               (state.states[n].velocity != None):
                                flight_parameters['velocity'] = np.append(
                                                    flight_parameters['velocity'],
                                                    state.states[n].velocity)
                                flight_parameters['climb_rate'] = np.append(
                                                    flight_parameters['climb_rate'],
                                                    state.states[n].vertical_rate)
                                flight_parameters['altitude'] = np.append(
                                                    flight_parameters['altitude'],
                                                    state.states[n].baro_altitude)
            return flight_parameters

        airframeDict = self._update_icao24s(self.csv_filepath, desired_airframes)

        icao24s = filter_icao24s(airframeDict[self.airframe]['icao24List'])

        # List of icao24s is converted to a dictionary (deals with case when
        #   length of list is greater than 500 elements)
        icao24s = configure_icao24s(icao24s)

        timestamp = self.timestamp
        flight_parameters = {'velocity':[], 'climb_rate':[], 'altitude':[]}

        while len(flight_parameters['velocity']) < num_data_points:
            for n in list(icao24s.keys()):
                print(timestamp, len(flight_parameters['velocity']))
                # Checks to see if airframe data is present at the current timestamp
                api = OpenSkyApi(api_username, api_password)
                state = api.get_states(time_secs=timestamp, icao24=icao24s[n])

                # If an airframe data is present at the current timestamp, sample
                #   surrounding timestamps for more airframe data.
                if state:
                    t1 = timestamp - ((time_increment-1)*60)
                    t2 = timestamp + ((time_increment-1)*60)
                    timestamp_list = np.linspace(t1, t2, (t2-t1)/60+1)
                    flight_parameters = scrape_opensky_data(timestamp_list,
                                                 icao24s[n], flight_parameters)

                if len(flight_parameters['velocity']) >= num_data_points:
                    break
            timestamp += time_increment*60

        # Storing values
        if save_data:
            icao24s = open(self.filepath + str(self.airframe) + '_' +
                           str(self.timestamp) + '.p', 'wb')
            pickle.dump(flight_parameters, icao24s)
            icao24s.close()
        self.velocity = np.array(flight_parameters['velocity'], dtype='float')
        self.climb_rate = np.array(flight_parameters['climb_rate'],
                                   dtype='float')
        self.altitude = np.array(flight_parameters['altitude'], dtype='float')

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
                 xgrid=np.linspace(0, 1, 400),
                 ygrid=np.linspace(0, 1, 400)):
        """Generate contour plot visualizing PDF of velocity vs. angle of
        attack."""

        X, Y = np.meshgrid(xgrid, ygrid)
        Z = np.exp(self.pdf.score_samples(np.array([X.ravel(), Y.ravel()]).T))
        Z = np.reshape(Z, X.shape)
        
        X, Y = self.denormalize(np.array([X, Y]).T).T
        
        plt.figure()
        plt.contourf(X, Y, Z)
        plt.xlabel(r'Angle of Attack ($^{\circ}$)')
        plt.ylabel('Velocity (m/s)')
        plt.colorbar(label='Probability')
        plt.xlim([min(X.ravel()), max(X.ravel())])
        plt.ylim([min(Y.ravel()), max(Y.ravel())])
        

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
        self.df = pd.DataFrame({'aoa': self.database[:,0], 'V': self.database[:,1]})
        self.df = self.df[(np.abs(stats.zscore(self.df)) < 2).all(axis=1)]
        print(min(self.df.values[:,0]),max(self.df.values[:,0]))
        print(min(self.df.values[:,1]),max(self.df.values[:,1]))
        
        self.database = self.normalize(self.df.values)
        self.pdf = KernelDensity(kernel='gaussian', bandwidth=0.01, algorithm='ball_tree').fit(self.database)

    def normalize(self, inputs):
        aoa, v = inputs.T
        aoa = (aoa - min(self.df.values[:,0]))/(max(self.df.values[:,0]) - min(self.df.values[:,0]))
        v = (v - min(self.df.values[:,1]))/(max(self.df.values[:,1]) - min(self.df.values[:,1]))
        return(np.array([aoa, v]).T)

    def denormalize(self, inputs):
        aoa, v = inputs.T
        aoa = aoa*(max(self.df.values[:,0]) - min(self.df.values[:,0])) + min(self.df.values[:,0])
        v =     v*(max(self.df.values[:,1]) - min(self.df.values[:,1])) + min(self.df.values[:,1])
        return(np.array([aoa, v]).T)