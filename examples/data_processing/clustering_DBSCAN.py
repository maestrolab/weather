import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.signal import savgol_filter

# additonal packages by Michael

import sklearn
import sklearn.utils
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# Colors for the clusters?
colors = [[0, 0.4470, 0.7410],
          [0.8500, 0.3250, 0.0980],
          [0.9290, 0.6940, 0.1250],
          [0.4940, 0.1840, 0.5560],
          [0.4660, 0.6740, 0.1880],
          [0.3010, 0.7450, 0.9330],
          [0.6350, 0.0780, 0.1840]]

# List used to cycle through each atmospheric profile of provided weather data
# 72797 removed

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

location = 0

# while location < len(locations):
while location < 1:
    
    if locations[location] == '72797':
        break
    
# Load atmospheric data ./../../../72469_profiles.p
# Elevation remains constant, there are 227 profiles
    directory = 'C:/Users/micha/Desktop/O-REU/Balloon Data/'
    data = pickle.load(open(directory+locations[location]+'.p', 'rb'))
    print(data.keys())
    # print('HUMIDITY:',data['humidity'][0],'\n')
    print('TEMPERATURE:',data['temperature'][0],'\n')
    print('HEIGHT:',data['height'],'\n')
    print('ELEVATION:', data['elevation'],'\n')
    
    # noises are single values
    
    print('NOISE:', data['noise'][0:10],'\n')
      
    # these lines only necessary for kmeans
    
    n = 200
    n_clusters = 5
    
    # only pulling humidity, take temperature, check order
    
    
    #############################
    
    
    rh = np.array(data['humidity'])
    # print(rh)
    temp = np.array(data['temperature'])
    m = len(rh)
    m_1 = len(temp)
    
    # creates and evenly spaced array from starting to end elevations with
    # n spacings (data points)
    alt_interpolated = np.linspace(data['elevation'][0], 13500, n)
    
    data_interpolated = np.zeros((len(rh), len(alt_interpolated), 2))
    for i in range(m):
        alt, values = np.array(rh[i]).T
        # values = savgol_filter(values, 11, 3)
        fun = interp1d(alt, values, fill_value="extrapolate")
        values_interpolated = fun(alt_interpolated)
        data_interpolated[i] = np.array([alt_interpolated, values_interpolated]).T
    
    
    #####################################
    
    # metrics
    average = np.array([np.average(data_interpolated[i, :, 1]) for i in range(m)])
    maximum = np.array([np.max(data_interpolated[i, :, 1]) for i in range(m)])
    ground = data_interpolated[:, 0, 1]
    indices_of_maximum = [np.argmax(data_interpolated[i, :, 1]) for i in range(m)]
    location_of_maximum = data_interpolated[i, indices_of_maximum, 0]
    average_profile = np.array([np.average(data_interpolated[0:m, j, 1]) for j in range(n)])
    standard = np.array(
        [np.average(np.absolute(data_interpolated[i, :, 1]-average_profile)) for i in range(m)])
    
    # clustering time
    points = data_interpolated[:, :, 1]
    
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(points)
    
    # standard scaler does some preprocessing
    
    points_1 = StandardScaler().fit_transform(points)
    # 10 and 20 get 2 clusters, only applicable to this file and still
    # has high noise
    db_scan = DBSCAN(eps = 10, min_samples = 20)
    
    # plot the 118 points with coloring from dbscan 
    
    # db points is an array, points_1 still in large dimensions
    
    db_points = db_scan.fit_predict(points_1)
    y_km = db_points
    
    # db_points is the reduced data
    
    # print('db_points=', db_points)
    labels = db_scan.labels_
    print('Unique numbers =/-1 belong to a cluster:', db_scan.labels_)

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Clusters =',n_clusters_)
    print('Noise =', n_noise_,'%')
    
    # need to specfiy color
    
    # only plotting a few columns, may not be necessary
    
    # keeps 1st and 2nd column
    plt.figure()
    plt.scatter(points_1[:,0], points_1[:,1], c=db_points)
    
    # dbscan and cluster, plot clusters against something else (colors as clusters)
    # pick and detect 2 other variables, plot against each other
    
    # leave this variable name the same
    
    # y_km = kmeans.fit_predict(points)
    
    # determine the centers of the clusters
    # what do the numerical values of centers and indexes mean?
    
    # centers = kmeans.cluster_centers_
    # centers = [np.average(centers[i]) for i in range(n_clusters)]
    # print('Centers =',centers)
    # print('Sort Centers =', (np.argsort(centers)))
    # indexes = np.arange(n_clusters)[np.argsort(centers)]
    indexes = np.arange(n_clusters_)
    print('indexes =', indexes)
    # print('Colors =',(np.array(colors)[indexes]))
    
    plt.figure()
    s = plt.scatter(average, location_of_maximum, c=data['noise'], cmap='gray')
    plt.colorbar(s)
    
    # noise vs. relative humidity
    plt.figure()
    for ii in indexes:
        plt.scatter(average[y_km == ii], np.array(data['noise'])[y_km == ii],
                    c=colors[ii], label=ii)
    plt.xlabel('Average Relative Humidity')
    plt.ylabel('Percieved Noise')
    # plt.legend()
    
    # average rh's vs. the high values rh's (higher average usually means high
    # humidities?)
    plt.figure()
    for ii in indexes:
        plt.scatter(average[y_km == ii], maximum[y_km == ii],
                    c=colors[ii], label=ii)
        
    # average rh vs. locations (tied to height/elevation?) lower rh = lower
    plt.figure()
    for ii in indexes:
        plt.scatter(average[y_km == ii], location_of_maximum[y_km == ii],
                    c=colors[ii], label=ii)
    
    # where max rh is in the array? vs. noise
    plt.figure()
    for ii in indexes:
        plt.scatter(location_of_maximum[y_km == ii], np.array(data['noise'])[y_km == ii],
                    c=colors[ii], label=ii)
    				
    plt.figure()
    plt.xlabel('Average')
    plt.ylabel('Noise')
    print('i', indexes)
    
    print(indexes[::-1])
    for ii in range(4):
        plt.scatter(average[y_km == ii], np.array(data['noise'])[y_km == ii],
                    c=colors[ii], label=ii)
        print(len(np.array(data['noise'])[y_km == ii]))
    
    plt.figure()
    for ii in indexes:
        plt.scatter(maximum[y_km == ii], np.array(data['noise'])[y_km == ii],
                    c=colors[ii], label=ii)
    
    # tSNE plot attempt
    
    sklearn.manifold.TSNE()
    
    # use points_1, now have reduced data after tsne, and make clusters colors 
    # from dbscan
    
    # learning rate too high = like a ball, too low = compressed w/ outliers
    data_tsne = TSNE(n_components = 2, perplexity = 30, 
                     learning_rate = 750).fit_transform(points_1)
    
    print(data_tsne.shape)
    
    # view the plot with certain parameters
    
    plt.figure()
    sns.scatterplot(data_tsne[:,0], data_tsne[:,1], hue = np.array(data['noise']))
    
    # plt.axis([-120, 120, -120, 120])

    print('Current file =', locations[location])
    
    # greater height = less rh
    print('max rh=',max(rh))
    # lower temps at higher elevations
    print('max temp=',max(temp))
    location += 1        
        
     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    # plt.legend()
    
    # plt.figure()
    # data = [data_interpolated[y_km == i] for i in range(n_clusters)]
    # average_plot = np.array([[np.average(data[i][:, :, 1], axis=0)] for i in range(n_clusters)])
    # for jj in indexes:
        # plt.subplot(2, 2, jj+1)
        # data_i = data_interpolated[y_km == jj]
        # for i in range(n):
            # x, y = data_i[i].T
            # plt.plot(y, x, jj, color='k', alpha=0.05)
        # plt.plot(average_plot[jj][0, :], data[jj][0, :, 0], color=colors[jj])