"""
Script to generate PDF of velocity vs. approximated angle of attack for
specified classes of aircraft.
"""
import numpy as np
from weather.scraper.path import Airframe

typecodeList = ['B737', 'B747', 'B757', 'B767', 'B777', 'B787',
                'A310', 'A318', 'A319', 'A320', 'A321',
                'A330', 'A340', 'A350', 'A380', 'C172', 'C180',
                'C182']

airFrame = Airframe(typecode=typecodeList[15], timestamp=1549036800)
# airFrame.update_icao24s()
# start_time = time.time()
# airFrame.update_OpenSkyApi() # must run for each typecode before pdf generation
# end_time = time.time()
# elapsed_time = end_time - start_time
# print('Elapsed Time = %.2f' % elapsed_time)
# pdf = airFrame.generate_pdf()
# print(pdf)

# np.linspace(np.amin(self.angleOfAttack), np.amax(self.angleOfAttack), 1000)
xgrid = np.linspace(-5, 35, 1000)
# np.linspace(np.amin(self.velocity), np.amax(self.velocity), 1000)
ygrid = np.linspace(20, 75, 1000)

X, Y = np.meshgrid(xgrid, ygrid)
parameters = np.vstack([X.ravel(), Y.ravel()])
pdf = airFrame.generate_pdf(parameters)
print(sum(pdf.ravel()))
airFrame.plot_pdf(parameters)
# airFrame.plot_scatter()
# airFrame.plot_weight()
