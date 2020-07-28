from rapidboom import AxieBump, EquivArea
from weather.boom import read_input
from weather.scraper.twister import process_data
# import tensorflow as tf
import platform
from weather.parameterize_atmosphere.autoencoder import *
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import csv
import subprocess as sp

##################################################################
mode = 't' # 't' for true and 'p' for predictions

if mode =='p':
    raw_data = pd.read_csv("./autoencoder_results_predictions.csv",skiprows=0) 
elif mode=='t':
    raw_data = pd.read_csv("./autoencoder_results_true.csv",skiprows=0)  
raw_data = pd.DataFrame.to_numpy(raw_data)
elevations = raw_data[:,3]

for profile in range(3593,len(elevations)):
    print("Profile: ",profile+1)
    f = open('profile.txt','w')
    f.write('%1.0f' % (profile)) 
    f.close()  
    
    f = open('mode.txt','w')
    f.write(mode) 
    f.close()  
    
    # run babysitter
    ps = sp.run('python ./sboom_babysitter.py', cwd = './', shell=True)