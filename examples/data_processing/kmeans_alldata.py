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

while location < len(locations):
    
    # initiate lists in here
    
    sub_height = []
    sub_temp = []
    sub_hum= []
    h_list = []
    temp_list = []
    hum_list = []
    
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
        # x_new = np.linspace(temp[i][0][0], 50000, 200) 
        x_new = np.linspace(5088, 50000, 200)
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
        # x1_new = np.linspace(temp[i][0][0], 50000, 200)   
        x1_new = np.linspace(5088, 50000, 200)
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
        
        # noise thing moved to here
        # this should completely change data
        
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
        sub_3 = []
        sub_3 = features + sub_1 + sub_2
        sub_1 = []
        sub_2 = []
        features = []
        main_1.append(sub_3)
    
    location += 1

#########################################
### CLUSTERING, SPECIFY CLUSTERS HERE ###
######################################### 

# normalize main data here

scaler = MinMaxScaler()
scaler.fit(main_1)
main_1 = scaler.transform(main_1)

points = np.array(main_1)

# iterate through different amount of clusters

n_clusters = [4, 8, 10, 12, 20]
# n_clusters = [8]

for m in range(len(n_clusters)):
    
    # reset the main lists every time it clusters (fixes graphs)
    n_list = []
    c_list= []
    
    kmeans = KMeans(n_clusters = n_clusters[m])   
    kmeans.fit(points)
    clusters = kmeans.cluster_centers_
    y_km = kmeans.fit_predict(points)
    
    # this line unscales the data after clustering for plotting
    # set to main_2, doesn't redefine main_1 and cause a spiral effect
    main_2 = scaler.inverse_transform(main_1)
    # main_2 = main_1
    
    # gets profiles to correspond to respective cluster, get noise associated
    # x is for clusters, i cycles through data for each of the days data was collected for
    for x in range(n_clusters[m]):
        for i in range(len(main_2)):
            if y_km[i] == x:
                 # organizes temp and hum data so indexes match cluster
                 subc_list.append(main_2[i])
                 # organizes noise data so indexes match cluster
                 subn_list.append(n1_list[i])
        c_list.append(subc_list)
        n_list.append(subn_list)
        subc_list = []
        subn_list = []
        
    # gets average temp and hum for each profile and associates with respective clusters
    # gets average for each cluster
    
    features = ['Average Temperature', 'Average Humidity', 'Max Humidity', 'Elevation']
    metrics = []
    main_maxh = []
    sub_maxh = []
    
    for x in range(n_clusters[m]):
         avg_tlist = []
         avg_hlist = []
         max_hum = 0
         for i in range(len(c_list[x])):
             avg_t = np.average(c_list[x][i][len(features):200+len(features)])
             avg_tlist.append(avg_t)
             avg_h = np.average(c_list[x][i][200+len(features):])
             avg_hlist.append(avg_h) 
             
             sub_maxh.append(max(c_list[x][i][200+len(features):]))
             
             # gets the max humidity for each cluster
             for z in range(200):
                 if c_list[x][i][z+200+len(features)] >= max_hum:
                     max_hum = c_list[x][i][z+200+len(features)]
         
         main_maxh.append(sub_maxh)
         sub_maxh = []
         avg_maxh = np.average(main_maxh[x])
                     
         avg_t_cl = np.average(avg_tlist)
         avg_h_cl = np.average(avg_hlist)
         avg_noise = np.average(n_list[x])
         std_t = np.std(avg_tlist) 
         std_h = np.std(avg_hlist)
         std_n = np.std(n_list[x])
   
         sub_metrics = []
         sub_metrics.append(x+1)
         sub_metrics.append(len(c_list[x]))
         sub_metrics.append(round(avg_t_cl, 2))
         sub_metrics.append(round(std_t, 2))
         sub_metrics.append(round(avg_h_cl, 2))
         sub_metrics.append(round(std_h, 2))
         sub_metrics.append(round(avg_noise, 2))
         sub_metrics.append(round(std_n, 2))
         sub_metrics.append(round(max_hum, 2))
         sub_metrics.append(round(avg_maxh, 2))
         
         metrics.append(sub_metrics)
         
    first = ['Cluster Number', 'Profiles in Cluster', 'Avg Temp', 'Std Temp', 'Avg Hum', 'Std Hum', 'Avg Noise',
            'Std Noise', 'Max Hum', 'Avg Max Hum']
    with open('data' + '.csv', mode='a+') as file:
        file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        file_writer.writerow(first)
        for i in range(len(metrics)):
            file_writer.writerow(metrics[i])
    
    ticks = []
    for i in range(n_clusters[m]+1):
        ticks.append(i-1)    
    
    plt.figure()
    sns.boxplot(data=n_list)
    # sns.swarmplot(data=n_list) # shows the datapoints
    plt.xticks(ticks=ticks)
    # plt.xlabel('Cluster Number')
    # plt.ylabel('Percieved Noise Level in PldB')   
    plt.title('Atmospheric Profile Noise Spread')
    plt.figure()
    sns.boxplot(data=n_list, showfliers=False)
    plt.xticks(ticks=ticks)
    plt.xlabel('Cluster Number')
    plt.ylabel('Percieved Noise Level in PldB')   
    plt.title('Atmospheric Profile Noise Spread')

    # 8 plots here: 2 temp spread (one w/outliers) 2 hum spread (one w/outliers)
    # 2 avg temp spread (one w/outliers) 2 avg hum spread (one w/outliers)
 
    ### LOTS OF PLOTTING ###
    ########################
 
    # tbox = []
    # tbox_main = []
    # for x in range(n_clusters[m]):
    #     for i in range(len(c_list[x])):
    #         tbox.append(c_list[x][i][3:203])
    #     tbox_main.append(tbox)
    #     tbox = []
        
    # plt.figure()
    # sns.boxplot(data=tbox_main)
    # plt.xticks(ticks=ticks)
    # plt.xlabel('Cluster Number')
    # plt.ylabel('Temperature in f')   
    # plt.title('Atmospheric Profile Temperature Spread')
    # plt.figure()
    # sns.boxplot(data=tbox_main, showfliers=False)
    # plt.xticks(ticks=ticks)
    # plt.xlabel('Cluster Number')
    # plt.ylabel('Temperature in f')   
    # plt.title('Atmospheric Profile Temperature Spread')
    
    # # average humidity spread for profiles in each cluster
    # hbox = []
    # hbox_main = []
    # for x in range(n_clusters[m]):
    #     for i in range(len(c_list[x])):
    #         hbox.append(c_list[x][i][203:403])
    #     hbox_main.append(hbox)
    #     hbox = []
    
    # plt.figure()
    # sns.boxplot(data=hbox_main)
    # plt.xticks(ticks=ticks)
    # plt.xlabel('Cluster Number')
    # plt.ylabel('Relative Humidity in %')   
    # plt.title('Atmospheric Profile Humidity Spread')
    # plt.figure()
    # sns.boxplot(data=hbox_main, showfliers=False )
    # plt.xticks(ticks=ticks)
    # plt.xlabel('Cluster Number')
    # plt.ylabel('Relative Humidity in %')   
    # plt.title('Atmospheric Profile Humidity Spread')
 
    tbox = []
    tbox_main = []
    for x in range(n_clusters[m]):
        for i in range(len(c_list[x])):
            tbox.append(np.average(c_list[x][i][3:203]))
        tbox_main.append(tbox)
        tbox = []
        
    plt.figure()
    sns.boxplot(data=tbox_main)
    plt.xticks(ticks=ticks)
    plt.xlabel('Cluster Number')
    plt.ylabel('Average Temperature in f')   
    plt.title('Atmospheric Profile Average Temperature Spread')
    plt.figure()
    sns.boxplot(data=tbox_main, showfliers=False)
    plt.xticks(ticks=ticks)
    plt.xlabel('Cluster Number')
    plt.ylabel('Average Temperature in f')   
    plt.title('Atmospheric Profile Average Temperature Spread')
    
    # average humidity spread for profiles in each cluster
    hbox = []
    hbox_main = []
    for x in range(n_clusters[m]):
        for i in range(len(c_list[x])):
            hbox.append(np.average(c_list[x][i][203:403]))
        hbox_main.append(hbox)
        hbox = []
    
    plt.figure()
    sns.boxplot(data=hbox_main)
    plt.xticks(ticks=ticks)
    plt.xlabel('Cluster Number')
    plt.ylabel('Average Humidity in %')   
    plt.title('Atmospheric Profile Average Humidity Spread')
    plt.figure()
    sns.boxplot(data=hbox_main, showfliers=False )
    plt.xticks(ticks=ticks)
    plt.xlabel('Cluster Number')
    plt.ylabel('Average Humidity in %')   
    plt.title('Atmospheric Profile Average Humidity Spread')

second = ['Profiles >=80 pLdB', '% of total', 'Outlier Profiles', 'Percentage of Total']
third = [loud_counter, round((loud_counter/profiles)*100, 2), outliers, round((outliers/profiles)*100, 2) ]
with open('data' + '.csv', mode='a+') as file:
    file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    file_writer.writerow(second)
    file_writer.writerow(third)
    


# print(len(main_1[0])) # 400 points in this index
# print(len(main_1))
# print(len(c_list[0])) # profiles in that respective cluster
# print(len(c_list[1]))
# print(len(c_list[0][0])) # 400 points here
# print(len(c_list)) # 10 clusters
# print(c_list[0][0][0])
# print(len(c_list)) # clusters
# print(len(c_list[0])) # profiles in that cluster
# print(c_list[0][0][:len(temp)]) # embedded: cluster, day, :len(temp)

### t-SNE plot ###

# sklearn.manifold.TSNE()
# data_tsne = TSNE(n_components = 2, perplexity = 50,  
#               learning_rate = 300).fit_transform(points) 
# plt.figure()
# # each point here represents 400 values: 200 temp and 200 hum
# sns.scatterplot(data_tsne[:,0], data_tsne[:,1], hue=np.array(n1_list))
# plt.title('%i ' % len(data_tsne) + 'Atmospheric Profiles with Noise')
# print('Dimensions of data_tsne, the amount of days:', len(data_tsne)) 
# print('Dimensions after being reduced:', len(data_tsne[0])) # is two because the dimensions were reduced
        
    



