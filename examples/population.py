from census import Census
from us import states
from aeropy.xfoil_module import output_reader

type_structure = ['string', 'string', 'string', 'float', 'float',
                  'float', 'float', 'float', 'float']
raw_data = output_reader('location.txt', type_structure=type_structure)
data_loc = {'id': raw_data['GEOID'], 'lat': raw_data['LAT'],
            'lon': raw_data['LON']}

property = 'B01003_001E'  # Total population
api_key = "d336cbb942af711df388ef67fda11759383df1a0"
c = Census(api_key)
data = c.acs5.state_county(property, '01', '001')

print(data_loc)
