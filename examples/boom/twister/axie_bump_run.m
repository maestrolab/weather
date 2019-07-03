function [perceived_loudness] = axie_bump_run(nBumps,bump_definition,atm,dirname)
% axie_bump_run takes the variables defining an axie-symmetric bump(s), writes these inputs to a text file, 
% and calls the python code that reads these inputs, runs Panair, sBOOM, and PyLdB to compute the perceived 
% loudness on the ground. Finally, this function reads the output text file to pass the loudness value
% back to Matlab.
%
% INPUTS:
%   nBumps - An integer describing how many bumps there are
%   bump_definition - a vector containing the geometries of the bumps
%       - This should be a vector of with 3*nBumps entries
%       - Every 3 entries should denote the height, length down body, and
%       width of a bump respectively
%   atm - a vector containing the day, month, year, hour, longitude,
%       latitude, and altitude (ft) to gather weather data
%
% EXAMPLE:
%   nBumps=2 % two bumps
%   bump_definition=[-0.006,10,1.3,-0.01,30,1.5] % geometries of bumps
%   atm = [18,6,2018,12,38,-107,45000] % [day, month, year, hour, latitude, longitude, altitude] 
%
%   This example prescribes the following two bumps:
%       1: height=-0.006, length down body=10, and width=1.3
%       2: height=-0.01, length down body=30, and width=1.5
%   at the atmospheric conditions on the 18th day of the 6th month of the
%   year 2018 at latitude and longitude of 38 and -107 degrees with the
%   plane at 45,000 feet.
%

%% Check inputs
if nargin <4
    dirname='';
    if nargin <3
        atm=[];
        if nargin <2
            error('axie_bump_run: Not enough inputs.')
        end
    end
end
% nBumps must be an integer
if length(nBumps)~=1 || floor(nBumps)~=nBumps
    error('axie_bump_run: ''nBumps'' must be an integer')
end
% bump_definition should be a vector of with 3*nBumps entries
if length(bump_definition)<(3*nBumps)
    error('axie_bump_run: There is not enough information in ''bump_definition'' to describe the number of bumps prescribed.')
elseif length(bump_definition)>(3*nBumps)
    warning('axie_bump_run: There is more information in ''bump_definition'' than is necessary to describe the number of bumps prescribed. Input 2 will be truncated to describe only the number of bumps prescribed.')
end
% the atmospheric data currently contains 7 entries
if ~isempty(atm) && length(atm)~=7
    error('axie_bump_run: The data in ''atm'' is incomplete or overdefined.')
end
% Allow dirname to also be a boolean, true
if isempty(dirname)
    dirname='';
elseif dirname % if it is a boolean true
    try
        w=getCurrentWorker; % get information about the current worker
        WorkerID = w.ProcessId; % get the worker's ID number
        dirname=strcat('dir_',num2str(WorkerID)); % set dirname equal to the directory corresponding to that worker
    catch % if no parallel pool is running, there is no worker ID
        dirname='';
    end
elseif ~isstring(dirname) % on the off chance that something is input other than a boolean true or a string
    dirname='';
end

%% Main Function
% Writing input file(s)
fileID = fopen(strcat('.\',dirname,'\axie_bump_inputs.txt'),'w'); % for parallel computing, dirname will contain the name of the directory corresponding to the current Matlab worker ID.
fprintf(fileID,'%8.6f\t',nBumps,bump_definition(1), bump_definition(2), bump_definition(3));
fclose(fileID);

counter=4;
for i=2:nBumps
    fileID = fopen(strcat('.\',dirname,'\axie_bump_inputs.txt'),'a');
    fprintf(fileID,'%8.6f\t',bump_definition(counter), bump_definition(counter+1), bump_definition(counter+2));
    fclose(fileID);
    counter=counter+3;
end

if ~isempty(atm)
    fileID = fopen(strcat('.\',dirname,'\axie_bump_atmsophere_inputs.txt'),'w');
    fprintf(fileID,'%8.6f\t',atm);
    fclose(fileID);
end

% Running Python command
if isempty(atm)
    %py.matlab_standard.run_main(dirname)% use the standard atmosphere
    if ~isempty(dirname)
        cd(strcat('.\',dirname));
    end
    system(strcat('python matlab_standard.py'));
else
    %py.matlab_real_weather.run_main(dirname)% use the prescribed atmosphere
    if ~isempty(dirname)
        cd(strcat('.\',dirname));
    end
    system(strcat('python matlab_real_weather.py'));
end

% Reading output file
fileID = fopen('.\axie_bump_outputs.txt','r');
% fileID = fopen('axie_bump_outputs.txt','r');
outputs = fscanf(fileID, '%f');
fclose(fileID);
perceived_loudness = outputs(1);

if ~isempty(dirname)
    cd ../
end
end
