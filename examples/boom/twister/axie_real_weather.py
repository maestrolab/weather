from rapidboom import AxieBump
from weather.scraper.twister import process_data
import platform

# Bump inputs
height = 0.1  # meters
length_down_body = 20  # meters
width = 6  # meters
bump_inputs = [height, length_down_body, width]

# Weather inputs
day = '18'
month = '06'
year = '2018'
hour = '12'
lat = 38
lon = -107
alt_ft = 45000

# Extracting data from database
alt_m = alt_ft * 0.3048
data, altitudes = process_data(day, month, year, hour, alt_m,
                               directory='../../../data/weather/twister/')
key = '%i, %i' % (lat, lon)
weather_data = data[key]

# Height to ground (HAG)
index = list(data.keys()).index(key)
height_to_ground = altitudes[index] / 0.3048

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
axiebump = AxieBump(CASE_DIR, PANAIR_EXE, SBOOM_EXE, altitude=height_to_ground,
                    weather=weather_data)
axiebump.MESH_COARSEN_TOL = 0.00045
axiebump.N_TANGENTIAL = 20
loudness = axiebump.run(bump_inputs)

print("Perceived loudness", loudness)
