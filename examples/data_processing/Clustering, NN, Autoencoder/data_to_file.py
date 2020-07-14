import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.signal import savgol_filter

# additonal packages by Michael

from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import sklearn
import seaborn as sns
from scipy import interpolate
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import csv

# what do these file names stand for?

locations = ['03953', '04220', '04270', '04360', '08508', '70133',
             '70200', '70219', '70231', '70261', '70273', '70316',
             '70326', '70350', '70398', '70414', '71043', '71081',
             '71109', '71119', '71600', '71603', '71722',
             '71811', '71815', '71816', '71823', '71836', '71845',
             '71867', '71906', '71907', '71908', '71909', '71913',
             '71925', '71926', '71934', '71945', '71957',
             '71964', '72201', '72202', '72206', '72208', '72210',
             '72214', '72215', '72230', '72233', '72235', '72240',
             '72248', '72249', '72251', '72261', '72265', '72274',
             '72293', '72305', '72317', '72318', '72327', '72340',
             '72357', '72363', '72364', '72365', '72376', '72388',
             '72393', '72402', '72403', '72426', '72440', '72451',
             '72456', '72469', '72476', '72489', '72493', '72501',
             '72518', '72520', '72528', '72558', '72562', '72572',
             '72582', '72597', '72632', '72634', '72645', '72649',
             '72659', '72662', '72672', '72681', '72694', '72712',
             '72747', '72764', '72768', '72776', '72786', '72797',
             '74005', '74389', '74455', '74494', '74560', '74646',
             '74794', '76256', '76394', '76458', '76526', '76595',
             '76612', '76644', '76654', '76679', '76805', '78016',
             '78073', '78384', '78397', '78486', '78526', '78583',
             '78807', '78897', '78954', '78970', '91285', '80222',
             '82022', '91165', '91285']

# need inside loop?
# sub = []
# main = []

# sub_height = []
# sub_temp = []
# sub_hum= []
# h_list = []
# temp_list = []
# hum_list = []

location = 0


# this list holds all profiles from every file
main_1 = []
# this list holds all profiles from every file corresponding to their
# respective cluster
# c_list = [] # moved to later on in the code

# loop list that resets every time it puts a cluster of profiles into c_list
subc_list = []
# loop list that resets every time it puts a cluster of noises into n1_list
subn_list = []
# this list holds the unclustered noise values
n1_list = []
# this list holds the clustered noise values for all profiles from every file
# n_list = [] # moved to later in the code
# how many noise profiles above 80 pLdB
loud_counter = 0
outliers = 0
profiles = 0
profile = 0


while location < len(locations):
    
    # initiate lists in here
    
    sub_height = []
    sub_temp = []
    sub_hum= []
    h_list = []
    temp_list = []
    hum_list = []
    hag_list = []
    
    current_file = locations[location]
    print('CURRENT FILE:', current_file)  
        
    directory = 'C:/Users/micha/Desktop/O-REU/Balloon Data/'
    data = pickle.load(open(directory+current_file+'.p', 'rb'))
    
    temp = (data['temperature'])
    hum = (data['humidity'])
    noise = (data['noise'])
    elevation = (data['elevation'])
 
    profiles += len(temp)   
 
    # max elevation from weather files == 5088.582677165354

    # noise into all one list
    
    # for i in range(len(noise)):
    #     n1_list.append(noise[i])
    #     if noise[i] >= 80:
    #         loud_counter += 1
        
    # x is day, i is data point within the day
    for x in range(len(temp)):
            for i in range(len(temp[x])):
                # gets all temp, hum, heights for each day in a seperated list
                sub_height.append(temp[x][i][0])
                sub_temp.append(temp[x][i][1])
                sub_hum.append(hum[x][i][1])  
            h_list.append(sub_height)
            temp_list.append(sub_temp)
            hum_list.append(sub_hum)
            sub_height = []
            sub_temp = []
            sub_hum= []    
            
     ### interpolation, extrapolation, and visualization
     # stands for interpolated and respective data value
    
    int_height = []
    int_temp = []
    int_hum = []
    for i in range(len(temp)):
        breaker = 0
        # temperature
        x = np.array(h_list[i])
        y = np.array(temp_list[i])
        f = interpolate.interp1d(x, y, fill_value = 'extrapolate')
        x_new = np.linspace(temp[i][0][0], 50000, 200) 
        # x_new = np.linspace(5088, 50000, 200)
        y_new = f(x_new)
        for m in range(len(y_new)):
            ## these prevent extreme extrapolation values 
            if y_new[m] <= -134:
                y_new[m] = -134
                breaker = 1
                outliers += 1
                break
            if y_new[m] >= 130:
                y_new[m] = 130
                breaker = 1
                outliers += 1
                break
        if breaker == 1:
            continue
        # humidity
        x_1 = np.array(h_list[i])
        y_1 = np.array(hum_list[i])
        f_1 = interpolate.interp1d(x_1, y_1, fill_value = 'extrapolate')
        x1_new = np.linspace(temp[i][0][0], 50000, 200)   
        # x1_new = np.linspace(5088, 50000, 200)
        y1_new = f_1(x1_new)
        for m in range(len(y1_new)):
            if y1_new[m] < 0:
                    y1_new[m] = 0
                    breaker = 1
                    outliers += 1
                    break
            if y1_new[m] > 100:
                    y1_new[m] = 100
                    breaker= 1
                    outliers += 1
                    break
        if breaker == 1:
            continue
        int_height.append(x_new)
        int_temp.append(y_new)
        int_hum.append(y1_new)
        
        # noise appending moved to here
        # this should completely change data
        # get first hag measurement for each profile
        hag_list.append(temp[i][0][0])
        n1_list.append(noise[i])
        if noise[i] >= 80:
            loud_counter += 1
        
        
        
    sub_1 = []
    sub_2 = []
    sub_3 = []
    features = [] # avg temp, avg hum, max hum
    
    int_height = list(int_height)
    int_temp = list(int_temp)
    int_hum = list(int_hum)
    
    # x is day i range is the 200 points
    # this loops gets all profiles into one big list
    # for x in range(len(temp)-outliers):
    for x in range(len(int_temp)):
        for i in range(len(int_temp[x])):
            sub_1.append(int_temp[x][i])
            sub_2.append(int_hum[x][i])
        features.append(np.average(int_temp[x]))
        features.append(np.average(int_hum[x]))
        features.append(max(int_hum[x]))
        ###
        features.append(elevation[0])
        features.append(hag_list[x])
        sub_3 = []
        # MAKE SURE RIGHT NOISE VALUE IS ADDED (done)
        sub_3 = features + sub_1 + sub_2 
        sub_1 = []
        sub_2 = []
        features = []
        main_1.append(sub_3)
    
    # don't forget to iterate the big while loop
    
    location += 1

# noise value in last column
for i in range(len(main_1)):
    main_1[i].insert(5, n1_list[i])

names = ['Average Temperature', 'Average Humidity', 'Max Humidity', 'ELevation', 'Min Hag', 'Noise']
for i in range(400):
    height_name = 'Height'
    names.append(height_name+str(i))
with open('autoencoder_data' + '.csv', mode='a+') as file:
    file_writer = csv.writer(file, delimiter=',', lineterminator='\n')
    file_writer.writerow(names)
    for i in range(len(main_1)):
        file_writer.writerow(main_1[i]) 
        




    
    
    
    
    
    
    
    
    
    
    
    