from rapidboom import AxieBump
from weather.boom import process_data, read_input
import platform

alt_ft = 45000.

# Collect input values
bump_inputs = read_input('axie_bump_inputs.txt')

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
# axiebump = AxieBump(CASE_DIR, PANAIR_EXE, SBOOM_EXE) # for standard atmosphere
axiebump = AxieBump(CASE_DIR, PANAIR_EXE, SBOOM_EXE, altitude=alt_ft)
axiebump.MESH_COARSEN_TOL = 0.00045
axiebump.N_TANGENTIAL = 20
loudness = axiebump.run(bump_inputs)

print("Perceived loudness", loudness)

# write output file for Matlab to read
f = open('axie_bump_outputs.txt', 'w')
f.write('%6.5f\t' % (loudness))
f.close()