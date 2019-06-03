from rapidboom import AxieBump, EquivArea
from weather.boom import read_input
from weather.scraper.twister import process_data
import platform

alt_ft = 50000.
run_method = 'EquivArea' # or 'panair'

# Collect input values
bump_inputs = read_input('axie_bump_inputs.txt', n=5)

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

if run_method == 'panair':
	# Run (Panair)
	axiebump = AxieBump(CASE_DIR, PANAIR_EXE, SBOOM_EXE, altitude=alt_ft, deformation='cubic')
	axiebump.MESH_COARSEN_TOL = 0.00045
	axiebump.N_TANGENTIAL = 20
	loudness = axiebump.run(bump_inputs)
elif run_method == 'EquivArea':
	# Run (equivalent area method)
	axiebump = EquivArea(CASE_DIR, SBOOM_EXE, altitude=alt_ft, deformation='cubic')
	loudness = axiebump.run(bump_inputs)
else:
	raise RuntimeError("evaluation method not recognized")

print("Perceived loudness", loudness)

# write output file for Matlab to read
f = open('axie_bump_outputs.txt', 'w')
f.write('%6.5f\t' % (loudness))
f.close()
