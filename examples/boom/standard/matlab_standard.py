from rapidboom import AxieBump, EquivArea
from weather.boom import read_input
from weather.scraper.twister import process_data
import platform
import os

try:
    FID = open("./eqarea_filename.txt","r")
    area_filename = FID.read()
    # delete file (eventually we can remove this, but this will help us keep consistent with older versions of this code)
    FID.close()
    os.remove("./eqarea_filename.txt")
except: # the file might not exist (this is a new feature)
    # use the "default" equivalent area distribution
    area_filename = 'x_59_ATA_dp_Pinf_vs_X_Probe5_trim_TRIMMED.eqarea'
print("Equivalent Area File: ",area_filename)

# define Mach number from eqarea_filename
try:
    # look for the word "Mach" in the filename
    index = area_filename.find("Mach")
    M = area_filename[index+4:] # remove everything before the word "Mach"
    # look for "_Alpha" in the filename
    index = M.find("_Alpha")
    M = M[:index] # remove everything after and including "_Alpha"
    # finally turn it into a number
    M = float(M)
    print("Mach detected from filename = {}".format(M))
except:
    M = 1.6 # 1.6 for 25D; 1.4 for X59
    print("Assuming Mach = {}".format(M))

# altitude of 25D is usually 50kft. In Lazzara SciTech2019 51.7kft was used. Altitude of X59 is 53.2kft.
alt_ft = 51700. # at some point, this should be an input too (along with Mach, phi, ref_length, and maybe r_over_l)
#atmosphere_input = './presb.input'
# run_method = 'EquivArea' # or 'panair'

# Collect input values
deformation, run_method, bump_inputs = read_input('axie_bump_inputs.txt')

CASE_DIR = "./"  # axie bump case
# PANAIR_EXE = 'panair.exe'  # name of the panair executable
# SBOOM_EXE = 'sboom_windows.dat.allow.exe'  # name of the sboom executable

print(platform.system())
if platform.system() == 'Linux' or platform.system() == 'Darwin':
    PANAIR_EXE = 'panair'
    SBOOM_EXE = 'sboom_linux'
elif platform.system() == 'Windows':
    PANAIR_EXE = 'panair.exe'
    SBOOM_EXE = 'sboom.exe'
else:
    raise RuntimeError("platfrom not recognized")

if run_method == 'panair':
    # Run (Panair)
    axiebump = AxieBump(CASE_DIR, PANAIR_EXE, SBOOM_EXE, altitude=alt_ft,
                        deformation=deformation)
    axiebump.MESH_COARSEN_TOL = 0.00045
    axiebump.N_TANGENTIAL = 20
    loudness = axiebump.run(bump_inputs)
elif run_method == 'EquivArea':
    # Run (equivalent area method)
    axiebump = EquivArea(CASE_DIR, SBOOM_EXE, altitude=alt_ft,
                         deformation=deformation,
                         area_filename = area_filename,
                         ref_length = 32.92, r_over_l = 5,  # for X59: ref_length = 27.432, for 25D: ref_length = 32.92,
                         mach= M, phi=0) # check Mach number and area filename every time!!!
    loudness = axiebump.run(bump_inputs)
else:
    raise RuntimeError("evaluation method not recognized")

print("Perceived loudness", loudness)

# write output file for Matlab to read
f = open('axie_bump_outputs.txt', 'w')
f.write('%6.5f\t' % (loudness))
f.close()
