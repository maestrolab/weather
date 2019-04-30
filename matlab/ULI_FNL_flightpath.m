% ULI_atmosphere_grib
% Chris Limbach, August 2018
% Laser Diagnostics and Plasma Devices Lab
% 
% This code pulls down GFS Analysis model atmospheric data from the NOAA 
% NOMADS server and interpolates and plots these parameters along 
% a given flight path. Specifically, temperature, wind velocity components,
% and humidity are considered.
%
% This code requires NCToolbox https://github.com/nctoolbox/nctoolbox
% NCToolbox provides functions for working with grib datasets.
% You may need to run the following command: run setup_nctoolbox
% clear all;


%%=================Specify Origin and Destination GPS Coordinates========%
%clear all;
%clc;
%Origin: Houston: +29.8156, 95.6737 W
%Destination: Seattle: 47.4502, 122.3088 W

lat_or=29.8156;             %latitude of origin
lat_dest=47.4502;           %latitude of destination
lon_or=360-95.6737;         %longitude of origin
lon_dest=360-122.3088;      %longitude of destination

Re=6378;                    %Earth Radius in km

%Distance between two points on the globe (spherical earth) "Great Arc"
flight_dist=Re*acos(sind(lat_or)*sind(lat_dest)+...
    cosd(lat_or)*cosd(lat_dest)*cosd(lon_dest-lon_or));

%Parameterized flight profile
t=linspace(0,1,100);                    %Parameter
flight_lat=lat_or+(lat_dest-lat_or)*t;  %Linear variation in latitude
flight_lon=lon_or+(lon_dest-lon_or)*t;  %Linear variation in longitude

%=====================Specify and Retrieve Data===========================%

%Date and Time of GFS Analysis Dataset
yr='2018';              %year
mo='05';                %month
day='01';               %Day
hr='0000';              % Time, Valid values: '0000', '0600', '1200', '1800'

%Check if grib data object (nco) already exists
%If not, download the dataset to "testgrib.grb2"
if(exist('nco') == 0)
url=['https://nomads.ncdc.noaa.gov/data/gfsanl/',yr,mo,'/',yr,mo,day,...
    '/gfsanl_4_',yr,mo,day,'_',hr,'_000.grb2']; 
outfilename = websave('testgrib.grb2',url);
end

%================Extract Key Parameters=====================%
% Extract key parameters from grib datafile
%outfilename='testgrib_full.grb2';

nco=ncgeodataset(outfilename);          %nco: geodataset object
param='Temperature_isobaric';
tempvar=nco.geovariable(param);         %extract temperature variable
param='Geopotential_height_isobaric';
altvar=nco.geovariable(param);          %extract geometric altitude (real space)
param='Relative_humidity_isobaric';
humvar=nco.geovariable(param);          %extract relative humidity
param='Vertical_velocity_pressure_isobaric';
%vel_r=nco.geovariable(param);           %extral vertical velocity
param='v-component_of_wind_isobaric'; %only 21 heights?
vel_v=nco.geovariable(param);           %extract v-component of wind
param='u-component_of_wind_isobaric';
vel_u=nco.geovariable(param);           %extract u-component of wind
param='Pressure_height_above_ground';
p_height=nco.geovariable(param);
param='isobaric';
p_h2=nco.geovariable(param);

%Other variables:
%param='Ozone_Mixing_Ratio_isobaric'; %only 17 heights?
%ozone=nco.geovariable(param);
%'u-component_of_wind_height_above_ground'
%'u-component_of_wind_altitude_above_msl'
%'Cloud_mixing_ratio_isobaric'
%'Pressure_height_above_ground'
param='Pressure_surface'; %(1x361x720)
p_surf=nco.geovariable(param); 

lat=nco{'lat'}(:);                      %extract latitude value vector
lon=nco{'lon'}(:);                      %extract longitude value vector
[Lat,Lon] = meshgrid(lat,lon);          %Generate the geophysical grid

%=====================Interpolate onto grid======================%

%Generate Grids               
TM=zeros(length(tempvar(1,:,1,1)),length(t)); %Temperture
UM=zeros(length(vel_u(1,:,1,1)),length(t)); %Velocity
VM=zeros(length(vel_v(1,:,1,1)),length(t)); %Velocity
RHM=zeros(length(humvar(1,:,1,1)),length(t)); %Relative humidity
AM=zeros(length(altvar(1,:,1,1)),length(t)); %Physical Altitude

if(length(tempvar(1,:,1,1)) ~= length(altvar(1,:,1,1)))
    display('Warning: Incomplete Dataset');
end

%Perform Geographic Interpolation at each flight altitude/isobar

for i=1:length(tempvar(1,:,1,1));

% extract temperature and physical altitude associate with
% each isobaric altitude
tempdata=tempvar.data(1,i,:,:);
altdata=altvar.data(1,i,:,:);
rhdata=humvar.data(1,i,:,:);
udata=vel_u.data(1,i,:,:);
vdata=vel_v.data(1,i,:,:);

% Generate geographic interpolants
FT=griddedInterpolant(Lon,-Lat,squeeze(tempdata)');
FA=griddedInterpolant(Lon,-Lat,squeeze(altdata)');
FH=griddedInterpolant(Lon,-Lat,squeeze(rhdata)');
FU=griddedInterpolant(Lon,-Lat,squeeze(udata)');
FV=griddedInterpolant(Lon,-Lat,squeeze(vdata)');

%Interpolate (linearly) onto the parameterized flight path.
TM(i,:)=FT(flight_lon,-flight_lat);
AM(i,:)=FA(flight_lon,-flight_lat);
UM(i,:)=FU(flight_lon,-flight_lat);
VM(i,:)=FV(flight_lon,-flight_lat);
RHM(i,:)=FH(flight_lon,-flight_lat);
end

%================Interpolate to Altitude Grid ======================%
% This part may fail on some days/times where, inexplicably, the dataset 
% does not reach this high of altitude

ALT_edge=linspace(0,18,19);         %vector of LIDAR altitude edges in km
ALT=(ALT_edge(1:end-1)+ALT_edge(2:end))/2;

%Interpolate onto the altitude grid
for i=1:length(t)
    Temp_ATM(:,i)=interp1(AM(:,i),TM(:,i),ALT*1000);
    Uvel_ATM(:,i)=interp1(AM(:,i),UM(:,i),ALT*1000);
    Vvel_ATM(:,i)=interp1(AM(:,i),VM(:,i),ALT*1000);
    RH_ATM(:,i)=interp1(AM(:,i),RHM(:,i),ALT*1000);
end

%Plot output
subplot(2,2,1);
[ch ch] = contourf(t*flight_dist,ALT,Temp_ATM,100); set(ch,'edgecolor','none'); colorbar;
subplot(2,2,2);
[ch ch] = contourf(t*flight_dist,ALT,RH_ATM,100); set(ch,'edgecolor','none'); colorbar;
subplot(2,2,3);
[ch ch] = contourf(t*flight_dist,ALT,Uvel_ATM,100); set(ch,'edgecolor','none'); colorbar;
subplot(2,2,4);
[ch ch] = contourf(t*flight_dist,ALT,Vvel_ATM,100); set(ch,'edgecolor','none'); colorbar;


%=====================Plot on United States====================%
%====================This Section is not working=================%