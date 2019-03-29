from rapidboom import AxieBump
from weather.boom import process_data
import platform

# bump design variables
# height = 0.1; # meters
# length_down_body = 20; # meters
# width = 6; # meters

# load weather data
day = '18'
month = '06'
year = '2018'
hour = '12'
lat = 38
lon = -107
alt_ft = 45000

# Extracting data from database
alt = alt_ft * 0.3048
data, altitudes = process_data(day, month, year, hour, alt,
                               directory='../data/weather/')
key = '%i, %i' % (lat, lon)
weather_data = data[key]

# Height to ground (HAG)
index = list(data.keys()).index(key)
height_to_ground = altitudes[index] / 0.3048

# Read inputs from a file
f = open('axie_bump_inputs.txt', 'r')
line = f.read()
line = line.split('\t')
f.close()

# Collect input values
inputs = []
for i in range(len(line)-1):
    inputs.append(float(line[i]))

n_bumps = inputs[0]  # this input will denote the number of bumps
bump_inputs = []  # initialize
if len(inputs) % 3 != 0:
    for i in range(1, len(inputs), 3):
        bump_inputs.append(inputs[i:i+3])
else:
    raise RuntimeError("The first input must denote the number of bumps")

CASE_DIR = "./"  # axie bump case
PANAIR_EXE = 'panair.exe'  # name of the panair executable
SBOOM_EXE = 'sboom_windows.dat.allow.exe'  # name of the sboom executable

print(platform.system())
if platform.system() == 'Linux':
    PANAIR_EXE = 'panair'
    SBOOM_EXE = 'sboom_linux'
elif platform.system() == 'Windows':
    PANAIR_EXE = 'panair.exe'
    SBOOM_EXE = 'sboom_windows.dat.allow'
else:
    raise RuntimeError("platfrom not recognized")

# Run
# axiebump = AxieBump(CASE_DIR, PANAIR_EXE, SBOOM_EXE) # for standard atmosphere
axiebump = AxieBump(CASE_DIR, PANAIR_EXE, SBOOM_EXE, altitude=height_to_ground,
                    weather=weather_data)
axiebump.MESH_COARSEN_TOL = 0.00045
axiebump.N_TANGENTIAL = 20
loudness = axiebump.run(bump_inputs)

print("Perceived loudness", loudness)

# write output file for Matlab to read
f = open('axie_bump_outputs.txt', 'w')
f.write('%6.5f\t' % (loudness))
f.close()
