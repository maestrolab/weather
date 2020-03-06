% clc; clear; close all; echo off;
% run setup_nctoolbox.m before anything
%% Dataset Inputs
yr = '2018';              %year
mo = '01';                %month
day = '02';               %Day
hr = '1200';              %Valid values: '0000', '0600', '1200', '1800'

min_lat = 13;
max_lat = 58;
min_lon = -144;
max_lon = -53;

step = 1.; %even multiples of .5
%% Convert longitude from (-180, 180) to (0,360)
if min_lon < 0
    min_lon = min_lon + 360;
end
if max_lon < 0
    max_lon = max_lon + 360;
end
%%
%Check if grib data object (nco) already exists
if exist('nco','var') == 0
url=['https://nomads.ncdc.noaa.gov/data/gfsanl/', yr, mo, '/', yr, mo, day,...
    '/gfsanl_4_', yr, mo, day, '_', hr, '_000.grb2']; 
outfilename = websave(strcat(yr, mo, day, '_', hr(1:2), '.grb'), url);
end

%% Extract key parameters from grib datafile

nco = ncgeodataset(outfilename);          %nco: geodataset object
temp = nco.geovariable('Temperature_isobaric'); % temperature (K)
hght = nco.geovariable('Geopotential_height_isobaric'); % height above sea level (m)
% hght = nco.geovariable('Geometrical_height'); % height above sea level (m)
relh = nco.geovariable('Relative_humidity_isobaric'); % relative humidity
vel_v = nco.geovariable('v-component_of_wind_isobaric'); % wind in y (m/s)
vel_u = nco.geovariable('u-component_of_wind_isobaric'); % wind in x (m/s)
p_height = nco.geovariable('Pressure_height_above_ground');

lat = nco.geovariable('lat');
lat = lat(:); % latitude (90,-90,0.5)
lon = nco.geovariable('lon');
lon = lon(:); % longitude (0,360,0.5)

%% Format data into usable form and export to excel and pickle if possible
N = length(temp(1,:,1,1)); % length of each array (number of points per variable)

% Data can be scraped for lat=90:-90) and for lon=(0,360) by 0.5 degrees

LAT = 2*(90-max_lat)+1 :2*step: 2*(90-min_lat)+1;
LON = (2*min_lon+1) :2*step: (2*max_lon+1);

s = struct();
s.height = hght(:, :, LAT, LON);
s.temperature = temp(:, :, LAT, LON);
s.wind_x = vel_u(:, :, LAT, LON);
s.wind_y = vel_v(:, :, LAT, LON);
s.humidity = relh(:, :, LAT, LON);
s.pressure = p_height(:, :, LAT, LON);
s.lat = lat(LAT).';
s.lon = lon(LON).';

% Convert longitude from (0,360) to (-180, 180)
s.lon(s.lon>180) = s.lon(s.lon>180) -360;
[s.lon_grid, s.lat_grid] = meshgrid(s.lon,s.lat);
Lon_flat = reshape(s.lon_grid.',1,[]);
Lat_flat = reshape(s.lat_grid.',1,[]);
s.lonlat = [Lon_flat(:), Lat_flat(:)];
filename = strcat(yr, mo, day, '_', hr(1:2),'.mat');
s=struct(s);
save(filename,'s')
disp('DONE')
