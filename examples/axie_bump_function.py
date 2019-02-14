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
alt_ft = 45000
alt = 51000 * 0.3048

data, altitudes = process_data(day, month, year, hour, alt,
                               directory='../data/weather/')
index = 2337  # index for worst case scenario
altitude = altitudes[index] / 0.3048
print(altitude)
key = list(data.keys())[index]
weather_data = data[key]


# Read inputs from a file
f = open('axie_bump_inputs.txt', 'r')
line = f.read()
line = line.split('\t')
f.close()

# Collect input values
inputs = []
for i in range(len(line)-1):
    inputs.append(float(line[i]))

height = inputs[0]
length_down_body = inputs[1]
width = inputs[2]

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
axiebump = AxieBump(CASE_DIR, PANAIR_EXE, SBOOM_EXE, altitude=altitude,
                    weather=weather_data)
axiebump.MESH_COARSEN_TOL = 0.00045
axiebump.N_TANGENTIAL = 20
loudness = axiebump.run([height, length_down_body, width])

print("Perceived loudness", loudness)

# write output file for Matlab to read
f = open('axie_bump_outputs.txt', 'w')
f.write('%6.5f\t' % (loudness))
f.close()
