from rapidboom import AxieBump, EquivArea
from weather.boom import read_input
from weather.scraper.twister import process_data
import platform


run_method = 'EquivArea'

# Bump design variables
bump_inputs = read_input('axie_bump_inputs.txt')

# Flight conditions inputs
#day = '18'
#month = '06'
#year = '2018'
#hour = '12'
#lat = 38
#lon = -107
#alt_ft = 45000.
f = open('axie_bump_atmsophere_inputs.txt', 'r')
line = f.read()
line = line.split('\t')
f.close()
# Collect input values
atm_inputs = []
for i in range(len(line)-1):
	atm_inputs.append(float(line[i]))
# Extract input values
if atm_inputs[0] < 10:
	day = "0" + str(int(atm_inputs[0]))
else:
	day = str(int(atm_inputs[0]))
if atm_inputs[1] < 10:
	month = "0" + str(int(atm_inputs[1]))
else:
	month = str(int(atm_inputs[1]))
year = str(int(atm_inputs[2]))
if atm_inputs[3] == 0:
	hour = "00"
elif atm_inputs[2] < 10:
	hour = "0" + str(int(atm_inputs[3]))
else:
	hour = str(int(atm_inputs[3]))
lat = atm_inputs[4]
lon = atm_inputs[5]
alt_ft = atm_inputs[6]

# Extracting data from database
alt_m = alt_ft * 0.3048
try:
	data, altitudes = process_data(day, month, year, hour, alt_m,
								   directory='../../../data/weather/twister/')
except(FileNotFoundError):
	data, altitudes = process_data(day, month, year, hour, alt_m,
								   directory='../../../../data/weather/twister/')
key = '%i, %i' % (lat, lon)
weather_data = data[key]

# Height to ground (HAG)
index = list(data.keys()).index(key)
height_to_ground = altitudes[index] / 0.3048

CASE_DIR = "./"  # axie bump case
PANAIR_EXE = 'panair.exe'  # name of the panair executable
SBOOM_EXE = 'sboom_windows.dat.allow.exe'  # name of the sboom executable

print(platform.system())
if platform.system() == 'Linux' or platform.system() == 'Darwin':
	PANAIR_EXE = 'panair'
	SBOOM_EXE = 'sboom_linux'
elif platform.system() == 'Windows':
	PANAIR_EXE = 'panair.exe'
	SBOOM_EXE = 'sboom_windows.dat.allow'
else:
	raise RuntimeError("platfrom not recognized")

# Run
if run_method == 'panair':
	axiebump = AxieBump(CASE_DIR, PANAIR_EXE, SBOOM_EXE,
                    altitude=height_to_ground,
                    weather=weather_data)
	axiebump.MESH_COARSEN_TOL = 0.00045
	axiebump.N_TANGENTIAL = 20
	loudness = axiebump.run(bump_inputs)
elif run_method == 'EquivArea':
	axiebump = EquivArea(CASE_DIR, SBOOM_EXE,
                    altitude=height_to_ground,
                    weather=weather_data)
	loudness = axiebump.run(bump_inputs)
else:
	raise RuntimeError("evaluation method not recognized")

print("Perceived loudness", loudness)

# write output file for Matlab to read
f = open('axie_bump_outputs.txt', 'w')
f.write('%6.5f\t' % (loudness))
f.close()
