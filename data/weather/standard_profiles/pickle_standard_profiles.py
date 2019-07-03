'''Script to generate a pickle file containing the standard atmospheric profiles
   for temperature and relative humidity.

   Temperature Profile:
    - https://www.engineeringtoolbox.com/standard-atmosphere-d_604.html

    Relative Humidity Profile:
     - Found in sBoom manual

   '''

import pickle
import xlrd as xr

# Load excel workbook
path = './Standard Profiles.xlsx'
book = xr.open_workbook(path)
sheets = {'temperature':book.sheet_by_index(0),
          'relative humidity':book.sheet_by_index(1),
          'pressure':book.sheet_by_index(2)}

# Store excel data into lists
standard_temperature = [[sheets['temperature'].cell_value(i,0),
                         sheets['temperature'].cell_value(i,1)]
                         for i in range(1,len(sheets['temperature'].col(0)))]

standard_relative_humidity = [[sheets['relative humidity'].cell_value(i,0),
                               sheets['relative humidity'].cell_value(i,1)]
                               for i in range(1,len(sheets['relative humidity']\
                               .col(0)))]

standard_pressure = [[sheets['pressure'].cell_value(i,0),
                      sheets['pressure'].cell_value(i,1)]
                      for i in range(1,len(sheets['pressure'].col(0))) if
                      sheets['pressure'].cell_value(i,0)!='']

data = {'temperature':standard_temperature,
        'relative humidity':standard_relative_humidity,
        'pressure':standard_pressure}

# Save lists in pickle file
standard_profiles = open('standard_profiles.p', 'wb')
pickle.dump(data, standard_profiles)
standard_profiles.close()
