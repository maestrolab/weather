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
from sklearn.preprocessing import StandardScaler

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

# all_data = list()
# plt.figure()
location = 0
# while location < len(locations):
while location < 1:
    
    current_file = locations[location]
    print('CURRENT FILE:', current_file)  

    if locations[location] == '72797':
        break
    
    # Load atmospheric data ./../../../72469_profiles.p
    # want to fix if data is the same for multiple files
    directory = 'C:/Users/micha/Desktop/O-REU/Balloon Data/'
    data = pickle.load(open(directory+current_file+'.p', 'rb'))
    # print(len(data))
    
    temp = (data['temperature'])
    hum = (data['humidity'])
    noise = (data['noise'])

    ####

    # all (height, temp) data in first index, eventually loop through all
    # indexes for this file
    
    #### 227 indexes per file ### CHANGE HERE ###
    
    # for i in range(len(data['temperature'])):
    for i in range(5):
        # print(len(data['temperature'])) # (227)
        print('CURRENT INDEX:', i)
        # important, goes through the 227 days for each file (line 86)
        
        # # for temp
        points = temp[i]
        points = np.array(points)
        
        # # for humidity
        points_1 = hum[i]
        points_1 = np.array(points_1)
        
        ### DBSCAN CLUSTERING ###
        
        # normalize the points (preprocessing) scale it back up after?
        points = StandardScaler().fit_transform(points)
        points_1 = StandardScaler().fit_transform(points_1)
        db = DBSCAN(eps = .3, min_samples = 7)
        db_temp = db.fit_predict(points)
        labels = db.labels_
        
        db = DBSCAN(eps = .3, min_samples = 7)
        db_hum = db.fit_predict(points_1)
        labels_1 = db.labels_
        
        n_clusters_temp = len(set(labels)) - (1 if -1 in labels else 0)
        n_clusters_hum = len(set(labels_1)) - (1 if -1 in labels_1 else 0)
        n_noise_ = list(labels).count(-1)
        n_noise_1 = list(labels_1).count(-1)
        
        print('Clusters Temp VS. Height =',n_clusters_temp)
        print('Noise Temp VS. Height =', n_noise_, '%')
        print('Clusters Hum VS. Height =',n_clusters_hum)
        print('Noise Hum VS. Height =', n_noise_1, '%')
     
        # Scaled temp vs. height   
     
        plt.figure()
        plt.scatter(points[:,0], points[:,1], c=db_temp)
        plt.xlabel('Normalized Height')
        plt.ylabel('Normalized Temperature')
        plt.title('DBSCAN Temperature VS. Height, File %i' % int(current_file) +', Index %i' % i) 
        
        # Scaled hum vs. height
        
        plt.figure()
        plt.scatter(points_1[:,0], points_1[:,1], c=db_hum)
        plt.xlabel('Normalized Height')
        plt.ylabel('Normalized Temperature')
        plt.title('DBSCAN Relative Humidity VS. Height, File: %i' % int(current_file) +', Index %i' % i)
        
        
        ### t-SNE plot ###
        
        # sklearn.manifold.TSNE()
        # data_tsne = TSNE(n_components = 2, perplexity = 30, 
        #              learning_rate = 100).fit_transform(points)
    
        # palette = sns.color_palette("bright", 10)
        # print(data_tsne)
        
        # need to associate noise with certain points
        
        # sns.scatterplot(data_tsne[:,0], data_tsne[:,1], hue=np.array(data['noise'][:len(points)]))
        # print(len(data_tsne))
        # plt.figure()
        # plt.scatter(data_tsne[:,0], data_tsne[:,1], c=np.array(data['noise'][:len(points)]))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    location += 1              
        
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        # plt.figure()
        # plt.scatter(height_list_temp, temp_list)
        # plt.xlabel('Height in ft.')
        # plt.ylabel('Temperature in f')
        # plt.figure()
        # plt.scatter(height_list_hum, hum_list)
        # plt.xlabel('Height in ft.')
        # plt.ylabel('Relative humidity in %')
        
        # plt.scatter(points[0][0][0], points[0][0][1])
        
        # now each (height, temp) from each index corresponds to each other
        # but are in separate lists; makes plotting easier?
        
        # shows that heights for different temps and humidities for same 
        # respective indexes are the same
              
        
               
                
                
                
        # PRINT STATEMENTS      
            
    # each data point index represents its respective data on a different day
    # print(data.keys())
    # print('(Month, Day):', data['month'], data['day'])      
    # print('Amount of same heights:', counter)       
    # print('Month list length:', len(data['month']))
    # print('Day list length:', len(data['day']))
    
    # figure out what wind represents
    # print(data['wind'])
    
    # each index has different (height in feet, temperature in f)
    # print(len(data['temperature'][1]))
    # print('INDEX BREAK')
    # print(data['temperature'][1])
    # print('INDEX BREAK')
    # print(data['temperature'][2])
    
    # print('temp_list:', temp_list)
    # print('height_list_temp:', height_list_temp)
    # print('hum_list:', hum_list)
    # print('height_list_hum:', height_list_hum)
    # print('length of temp_list:', len(temp_list))
    # print('length of height_list_temp:', len(height_list_temp))
    # print('length of hum_list:', len(hum_list))
    # print('length of height_list_hum:', len(height_list_hum))  
    
    # each noise can correspond to multiple data points?
    # print('Length of noise data:', len(data['noise']))
    # print('Noise', data['noise'])
    
    # after running all files
    # length of temp_list: 18115851
    # length of height_list_temp: 18115851
    # length of hum_list: 18116018
    # length of height_list_hum: 18116018
    # Length of noise data: 436




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    