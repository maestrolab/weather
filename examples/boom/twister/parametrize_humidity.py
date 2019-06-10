import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from examples.boom.twister.misc_humidity import HermiteSpline, SplineBumpHumidity

class ParametrizeHumidity:

    def __init__(self, altitudes, relative_humidities, temperatures,
                 pressures, bounds, parametrize_altitudes = np.array([]),
                 geometry_type = 'spline'):
        self.alts = altitudes
        self.rhs = relative_humidities
        self.temps = temperatures
        self.pressures = pressures
        self._calculate_vapor_pressures()
        self.geometry_type = geometry_type
        self.bounds = bounds

        if parametrize_altitudes.any():
            self.p_alts = parametrize_altitudes
        else:
            self.p_alts = self.alts[:]

    def _calculate_vapor_pressures(self):
        saturation_vapor_pressures = []
        for i in range(len(self.temps)):
            if self.temps[i]>=0:
                f = 1.0007+(3.46e-6*self.pressures[i])
                sat_vps = f*0.61121*np.exp(17.502*self.temps[i]/
                          (240.97+self.temps[i]))
            elif self.temps[i]>-50:
                f = 1.0003+(4.18e-6*self.pressures[i])
                sat_vps = f*0.61115*np.exp(22.452*self.temps[i]/
                          (272.55+self.temps[i]))
            else:
                f = 1.0003+(4.18e-6*self.pressures[i])
                sat_vps = f*0.61115*np.exp(22.542*self.temps[i]/
                          (273.48+self.temps[i]))
            saturation_vapor_pressures.append(sat_vps)

        actual_vapor_pressures = [self.rhs[i]/100*
                                  saturation_vapor_pressures[i] for i in
                                  range(len(self.rhs))]
        self.saturation_vps = saturation_vapor_pressures[:]
        self.vps = actual_vapor_pressures[:]

    def normalize_inputs(self, inputs, inverse = False):
        outputs = np.zeros(len(inputs))
        bounds = self.bounds[:]
        if inverse:
            for i in range(len(inputs)):
                outputs[i] = (inputs[i]-bounds[i][0])/(bounds[i][1]-bounds[i][0])
        else:
            for i in range(len(inputs)):
                outputs[i] = inputs[i]*(bounds[i][1]-bounds[i][0]) + bounds[i][0]
        return outputs

    def geometry(self, x):
        if self.geometry_type == 'spline':
            p0,m0,m1,b = x
            self.spline = HermiteSpline(p0=p0,m0=m0,m1=m1,b=b,p1=0,a=0)
            self.p_vps = self.spline(parameter=np.array(self.p_alts))
        elif self.geometry_type == 'CST':
            A = list(x[:-2])
            A.append(0)
            chord = max(self.p_alts)
            deltaLE = x[-1]
            self.p_vps = CST(altitude, Au=A, deltasLE=0, N1=0, N2=1, c= chord, deltasz=0)
        elif self.geometry_type == 'spline_bump':
            p0,m0,m1,m,i,y,b = x
            index = int(np.round(i))
            a = 0
            p1 = 0
            x = self.p_alts[index]
            self.spline = SplineBumpHumidity(a=a, b=b, x=x, p0=p0, p1=p1, y=y,
                                             m0=m0, m1=m1, m=m)
            self.p_vps = self.spline(parameter=np.array(self.p_alts))

    def calculate_humidity_profile(self):
        self.p_rhs = [100*self.p_vps[i]/self.saturation_vps[i] for i in range(
                                                               len(self.p_vps))]

    def RMSE(self, x):
        x = self.normalize_inputs(x)
        self.geometry(x)
        p_vps_compare = np.interp(self.alts, self.p_alts, self.p_vps)
        self.rmse = mean_squared_error(self.vps, p_vps_compare)
        if self.rmse < 0.01:
            print(self.rmse)
        return self.rmse

    def plot(self, profile_type='vapor_pressures'):
        fig = plt.figure()
        if profile_type == 'vapor_pressures':
            plt.plot(self.vps, self.alts, label='Original')
            plt.plot(self.p_vps, self.p_alts, label='Parametrized')
            plt.xlabel('Vapor Pressure [kPa]')
        elif profile_type == 'relative_humidities':
            plt.plot(self.rhs, self.alts, label='Original')
            plt.plot(self.p_rhs, self.p_alts, label='Parametrized')
            plt.xlabel('Relative Humidities [%]')
        plt.ylabel('Altitude [m]')
        plt.legend()

    def plot_percent_difference(self, i):
        i = int(np.round(i))
        num_ = abs(self.vps[:i]-self.p_vps[:i])
        den = (self.vps[:i]+self.p_vps[:i])/2
        percent_diff = num_/den*100

        fig = plt.figure()
        plt.plot(percent_diff, self.p_alts[:i])
        plt.xlabel('Percent Difference [%]')
        plt.ylabel('Altitude [m]')
