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

# loop to iterate throughout all weather data, see which files missing

location = 0

# a loop for all data and a loop for a single file for testing code

# while location < len(locations):
while location < 1:

    if locations[location] == '72797':
        break
    
# Load atmospheric data ./../../../72469_profiles.p

    directory = 'C:/Users/micha/Desktop/O-REU/Balloon Data/'
    data = pickle.load(open(directory+locations[0]+'.p', 'rb'))
    
    # review dicts; elevations constant?
    print(data.keys())
    print('ELEVATION:', data['elevation'][0:10])
    print('length elevation:', len(data['elevation']))
    print('HEIGHT:',data['height'][0:9])
    print('TEMPERATURE:', data['temperature'][0])
    # print('HUMIDITY:', data['humidity'][0:9])    
    n = 200
    n_clusters = 4
    
    # 227 (depends on file?) different indexes per file, each index has around 40 - 60 2 dimensional 
    # data points in the format (height, temperature)
    # unable to collect data for all heights, so interpolated on line 95
    # elevations are all the same, even for different files (98.42)
    # heights are what are different
    print('length:', len(data['temperature'][0]))
    temp_list = list()
    for i in range(len(data['temperature'][0])):
        # list of just the temps at varying heights
        temp_list.append(data['temperature'][0][i][1])
    print('length temp_list', len(temp_list))
    print('temp_list:', temp_list)
    
    # only taking in relative humidity data
    rh = np.array(data['humidity'])
    m = len(rh)
    alt_interpolated = np.linspace(data['elevation'][0], 13500, n)
    
    data_interpolated = np.zeros((len(rh), len(alt_interpolated), 2))
    for i in range(m):
        alt, values = np.array(rh[i]).T
        # values = savgol_filter(values, 11, 3)
        fun = interp1d(alt, values, fill_value="extrapolate")
        values_interpolated = fun(alt_interpolated)
        data_interpolated[i] = np.array([alt_interpolated, values_interpolated]).T
    
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
    points = data_interpolated[:, :, 1] # all, all, only first index (temp)
    
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(points)
    
    y_km = kmeans.fit_predict(points)
    
    centers = kmeans.cluster_centers_
    centers = [np.average(centers[i]) for i in range(n_clusters)]
    print('Centers =',centers)
    print('Sort Centers =', (np.argsort(centers)))
    indexes = np.arange(n_clusters)[np.argsort(centers)]
    print('indexes =', indexes)
    print('Colors =',(np.array(colors)[indexes]))
    
    plt.figure()
    s = plt.scatter(average, location_of_maximum, c=data['noise'], cmap='gray')
    plt.colorbar(s)
    
    plt.figure()
    for ii in indexes:
        plt.scatter(average[y_km == ii], np.array(data['noise'])[y_km == ii],
                    c=colors[ii], label=ii)
    # plt.legend()
    
    plt.figure()
    for ii in indexes:
        plt.scatter(average[y_km == ii], maximum[y_km == ii],
                    c=colors[ii], label=ii)
    
    plt.figure()
    for ii in indexes:
        plt.scatter(average[y_km == ii], location_of_maximum[y_km == ii],
                    c=colors[ii], label=ii)
    
    plt.figure()
    for ii in indexes:
        plt.scatter(location_of_maximum[y_km == ii], np.array(data['noise'])[y_km == ii],
                    c=colors[ii], label=ii)
    				
    plt.figure()
    plt.xlabel('Average')
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
    
    # t-SNE plot 
    
    sklearn.manifold.TSNE()
    
    # tried 'data' then 'points' worked after getting an error with data
    # perplexity default = 30
    # color based on parameters
    
    data_tsne = TSNE(n_components = 2, perplexity = 30, 
                     learning_rate = 750).fit_transform(points)
    
    print(data_tsne.shape)
    
    # palette = sns.color_palette("bright", 10)
    
    plt.figure()
    sns.scatterplot(data_tsne[:,0], data_tsne[:,1], hue = np.array(data['noise']))
    
    # plot with different ranges, or can use console
    plt.axis([-500, 500, -500, 500])
    
    # continue the iteration until the final index of the file list is reached
    
    print('Current file =', locations[location])
    location += 1        
        
    plt.show()  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
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