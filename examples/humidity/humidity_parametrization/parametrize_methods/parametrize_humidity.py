import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d

from misc_humidity import HermiteSpline, SplineBumpHumidity


class ParametrizeHumidity:

    def __init__(self, altitudes, relative_humidities, temperatures,
                 pressures, bounds, parametrize_altitudes=np.array([]),
                 geometry_type='spline'):
        self.alts = np.array(altitudes)
        self.rhs = relative_humidities
        self.temps = temperatures
        self.pressures = pressures
        self._calculate_vapor_pressures()
        self.geometry_type = geometry_type
        self.bounds = bounds

        if parametrize_altitudes.any():
            self.p_alts = np.array(parametrize_altitudes)
        else:
            self.p_alts = np.array(self.alts[:])

    def _calculate_vapor_pressures(self):
        '''Arden Buck Equations (1981)'''
        saturation_vapor_pressures = []
        for i in range(len(self.temps)):
            if self.temps[i] >= 0:
                f = 1.0007+(3.46e-6*self.pressures[i])
                sat_vps = f*0.61121*np.exp(17.502*self.temps[i] /
                                           (240.97+self.temps[i]))
            elif self.temps[i] > -50:
                f = 1.0003+(4.18e-6*self.pressures[i])
                sat_vps = f*0.61115*np.exp(22.452*self.temps[i] /
                                           (272.55+self.temps[i]))
            else:
                f = 1.0003+(4.18e-6*self.pressures[i])
                sat_vps = f*0.61115*np.exp(22.542*self.temps[i] /
                                           (273.48+self.temps[i]))
            saturation_vapor_pressures.append(sat_vps)

        actual_vapor_pressures = [self.rhs[i]/100 *
                                  saturation_vapor_pressures[i] for i in
                                  range(len(self.rhs))]
        self.saturation_vps = np.array(saturation_vapor_pressures[:])
        self.vps = np.array(actual_vapor_pressures[:])

    def normalize_inputs(self, inputs, inverse=False):
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
            p0, p1, m0, m1, b = x
            self.spline = HermiteSpline(p0=p0, m0=m0, m1=m1, b=b, p1=p1, a=0)
            self.p_vps = self.spline(parameter=np.array(self.p_alts))
            # All points above greater than b will be assigned to the value of p1
            indexes = np.where(np.array(self.p_alts) > b)
            self.p_vps[indexes] = p1
        elif self.geometry_type == 'CST':
            A = list(x[:-2])
            A.append(0)
            chord = max(self.p_alts)
            deltaLE = x[-1]
            self.p_vps = CST(altitude, Au=A, deltasLE=0, N1=0, N2=1, c=chord, deltasz=0)
        elif self.geometry_type == 'spline_bump':
            p0, p1, m0, m1, m, x, y, b = x
            a = 0
            self.spline = SplineBumpHumidity(a=a, b=b, x=x, p0=p0, p1=p1, y=y,
                                             m0=m0, m1=m1, m=m)
            self.p_vps = self.spline(parameter=np.array(self.p_alts))
            # All points above greater than b will be assigned to the value of p1
            indexes = np.where(np.array(self.p_alts) > b)
            self.p_vps[indexes] = p1
        elif self.geometry_type == 'log':
            a, b = x
            # Need to fix; not a good method (but linear fit in log domain does
            #   not appear to be the best option for parametrization so may not
            #   correct)
            self.p_alts[0] = 1
            self.p_vps = a*np.log(self.p_alts)+b
        elif self.geometry_type == 'spline_log':
            p0, p1, m0, m1, b, a = x
            self.spline = HermiteSpline(p0=p0, p1=p1, m0=m0, m1=m1, b=b, a=a)
            log_alts = np.append(a, np.log(self.p_alts[1:]))
            self.p_vps = self.spline(parameter=log_alts)
        elif self.geometry_type == 'spline_bump_log':
            p0, p1, m0, m1, m, x, y, a, b = x
            self.spline = SplineBumpHumidity(a=a, b=b, x=x, p0=p0, p1=p1, y=y,
                                             m0=m0, m1=m1, m=m)
            log_alts = np.append(a, np.log(self.p_alts[1:]))
            self.p_vps = self.spline(parameter=log_alts)

    def calculate_humidity_profile(self):
        self.p_rhs = [100*self.p_vps[i]/self.saturation_vps[i] for i in range(
            len(self.p_vps))]

    def RMSE(self, x, profile_type='vapor_pressures', print_rmse=False):
        x = self.normalize_inputs(x)
        self.geometry(x)
        if profile_type == 'vapor_pressures':
            if (self.alts == self.p_alts).any():
                self.rmse = mean_squared_error(self.vps, self.p_vps)
            else:
                p_vps_compare = np.interp(self.alts, self.p_alts, self.p_vps)
                self.rmse = mean_squared_error(self.vps, p_vps_compare)
        elif profile_type == 'relative_humidities':
            self.calculate_humidity_profile()
            self.rmse = mean_squared_error(self.rhs, self.p_rhs)
        if print_rmse:
            print(self.rmse)
        return self.rmse

    def plot(self, profile_type='vapor_pressures'):
        # fig = plt.figure()
        if profile_type == 'vapor_pressures':
            plt.plot(self.vps, self.alts, label='Original')
            plt.scatter(self.vps, self.alts, label='Original')
            plt.plot(self.p_vps, self.p_alts, label='Parametrized')
            plt.xlabel('Vapor Pressure [kPa]')
            plt.ylabel('Altitude [m]')
        elif profile_type == 'relative_humidities':
            plt.plot(self.rhs, self.alts, label='Original')
            plt.scatter(self.rhs, self.alts, label='Original')
            plt.plot(self.p_rhs, self.p_alts, label='Parametrized')
            plt.xlabel('Relative Humidities [%]')
            plt.ylabel('Altitude [m]')
        elif profile_type == 'log':
            plt.plot(self.vps[1:], np.log(self.alts[1:]), label='Original')
            plt.scatter(self.vps[1:], np.log(self.alts[1:]), label='Original')
            plt.plot(self.p_vps[1:], np.log(self.p_alts[1:]), label='Parametrized')
            plt.xlabel('Vapor Pressure [kPa]')
            plt.ylabel('log(Altitude)')
        plt.legend()
