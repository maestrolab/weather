function [perceived_loudness] = axie_bump_run(nBumps,bump_definition,bump_type,run_method,atm,dirname)
% axie_bump_run takes the variables defining an axie-symmetric bump(s), writes these inputs to a text file, 
% and calls the python code that reads these inputs, runs Panair, sBOOM, and PyLdB to compute the perceived 
% loudness on the ground. Finally, this function reads the output text file to pass the loudness value
% back to Matlab.
%
% INPUTS:
%   nBumps: An integer describing how many bumps there are
%   bump_definition: a vector containing the geometries of the bumps
%       - This should be a vector of with N*nBumps entries 
%       - N is dependent on the bump_type (N=3 for gaussian bump; N=5 for
%           cubic spline bump) 
%           - Gaussian bump: every 3 entries should denote the height,
%              length down body, and width of a bump respectively
%           - Cubic spline bump: every 5 entries should denote the
%              location, height, slope, left width, right width of a bump
%              respectively  
%           - Corrected cubic spline bump: every 4 entries should denote
%              the location, height, left width, right width of a bump
%              respectively (slope=0)
%   bump_type: a string denoting which type of deformation we will consider
%       - Available options: 'gaussian' or 'cubic' or 'cubic_cor' (neglects
%           slope) 
%   run_method: a string denoting which method will be used to calculate
%       the near-field pressure signature 
%       - Available options: 'panair' or 'EquivArea'
%   atm: Either
%       - an empty vector to use the standard atmospheric conditions, OR
%       - a vector containing the day, month, year, hour, longitude,
%           latitude, and altitude (ft) to gather weather data (7 entries),
%           OR
%       - a boolean true value indicating that the weather data is already
%           saved in the current directory in the sBOOM compatible file:
%           'presb.input', OR
%       - a vector containing the 3-5 parameters used to generate a weather
%           profile based on the autoencoder
%   dirname: a boolean or string denoting the location to work in
%       - if a string is given, the working directory will be set using
%         this location (this location will contain all input/output files
%         used during this process)
%       - if a boolean 'true' is given, the function will attempt to access
%         a directory named by the current parallel worker's ID number (if it
%         fails to access this directory either because it does not exist or
%         there is no parallel worker, the current working directory will be
%         used)
%       - if the anything else is input, the current working directory will
%         be used
%
% EXAMPLE:
%   nBumps=2 % two bumps
%   bump_definition=[-0.006,10,1.3,-0.01,30,1.5] % geometries of bumps
%   bump_type='gaussian'
%   run_method='panair'
%   atm = [18,6,2018,12,38,-107,45000] % [day, month, year, hour, latitude, longitude, altitude] 
%   dirname='./mydir'
%
%   This example prescribes the following two gaussian bumps:
%       1: height=-0.006, length down body=10, and width=1.3
%       2: height=-0.01, length down body=30, and width=1.5
%   at the atmospheric conditions on the 18th day of the 6th month of the
%   year 2018 at latitude and longitude of 38 and -107 degrees with the
%   plane at 45,000 feet. It will call PANIAR to generate the nearfield
%   pressure distribution and read/write all input/output files to a
%   directory called 'mydir' contained within the present working
%   directory.
%
% Notes:
%   The working directory needs to contain the main python file. Note that
%   matlab_standard.py is located in "weather\examples\boom\standard" while
%   matlab_real_weather.py is located in "weather\examples\boom\twister".
%   Therefore, the working directory needs to be changed according to which
%   case is to be run.

%% Check inputs
if nargin <6
    dirname=[];
    if nargin <5
        atm=[];
        if nargin <4
            run_method = [];
            if nargin <3
                bump_type=[];
                if nargin <2
                    error('ERROR: axie_bump_run: Not enough inputs.')
                end
            end
        end
    end
end

% nBumps must be an integer
if length(nBumps)~=1 || floor(nBumps)~=nBumps
    error('ERROR: axie_bump_run: ''nBumps'' must be an integer')
end
% bump_type must match the available types
if isempty(bump_type)
    bump_type='gaussian'; % default bump type
end
% length of bump_definition is dependent on bump_type
switch bump_type
    case 'gaussian'
        % bump_definition should be a vector of with 3*nBumps entries
        if length(bump_definition)<(3*nBumps)
            error('ERROR: axie_bump_run: There is not enough information in ''bump_definition'' to describe the number of bumps prescribed.')
        elseif length(bump_definition)>(3*nBumps)
            warning('WARNING: axie_bump_run: There is more information in ''bump_definition'' than is necessary to describe the number of bumps prescribed. Input 2 will be truncated to describe only the number of bumps prescribed.')
        end
        deformation_flag=1;
    case 'cubic'
        % bump_definition should be a vector of with 5*nBumps entries
        if length(bump_definition)<(5*nBumps)
            error('ERROR: axie_bump_run: There is not enough information in ''bump_definition'' to describe the number of bumps prescribed.')
        elseif length(bump_definition)>(5*nBumps)
            warning('WARNING: axie_bump_run: There is more information in ''bump_definition'' than is necessary to describe the number of bumps prescribed. Input 2 will be truncated to describe only the number of bumps prescribed.')
        end
        deformation_flag=2;
    case 'cubic_cor'
        % bump_definition should be a vector of with 5*nBumps entries
        if length(bump_definition)<(4*nBumps)
            error('ERROR: axie_bump_run: There is not enough information in ''bump_definition'' to describe the number of bumps prescribed.')
        elseif length(bump_definition)>(4*nBumps)
            warning('WARNING: axie_bump_run: There is more information in ''bump_definition'' than is necessary to describe the number of bumps prescribed. Input 2 will be truncated to describe only the number of bumps prescribed.')
        end
        deformation_flag=2;
    otherwise
        error('ERROR: axie_bump_run: That bump_type case is not permitted.')
end
% run_method must match available types
if isempty(run_method)
    run_method='EquivArea';
end
switch run_method
    case 'panair'
        run_method_flag=1;
    case 'EquivArea'
        run_method_flag=2;
    otherwise
        error(strcat('ERROR: axie_bump_run: That ''run_method'' case is not permitted. Please use one of the available cases: ',...
            '''panair'' or ''EquivArea''.'))
end
% the atmospheric data currently contains 7 entries
if isempty(atm)
elseif length(atm)>7
    error('ERROR: axie_bump_run: The data in ''atm'' is overdefined.')
elseif length(atm)==2 || length(atm)==6
    error('ERROR: axie_bump_run: The data in ''atm'' is incorrectly defined.')
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
fprintf(fileID,'%8.6f\t',nBumps,deformation_flag,run_method_flag);
switch bump_type
    case 'gaussian'
        fprintf(fileID,'%8.6f\t',bump_definition(1), bump_definition(2), bump_definition(3));
        counter=4;
        for i=2:nBumps
        fprintf(fileID,'%8.6f\t',bump_definition(counter), bump_definition(counter+1), bump_definition(counter+2));
        counter=counter+3;
        end
    case 'cubic'
        fprintf(fileID,'%8.6f\t',bump_definition(1), bump_definition(2), bump_definition(3), bump_definition(4), bump_definition(5));
        counter=6;
        for i=2:nBumps
        fprintf(fileID,'%8.6f\t',bump_definition(counter), bump_definition(counter+1), bump_definition(counter+2), bump_definition(counter+3), bump_definition(counter+4));
        counter=counter+5;
        end
    case 'cubic_cor'
        fprintf(fileID,'%8.6f\t',bump_definition(1), bump_definition(2), 0, bump_definition(3), bump_definition(4));
        counter=5;
        for i=2:nBumps
        fprintf(fileID,'%8.6f\t',bump_definition(counter), bump_definition(counter+1),0, bump_definition(counter+2), bump_definition(counter+3));
        counter=counter+4;
        end
end
fclose(fileID);

if ~isempty(atm)
    fileID = fopen(strcat('.\',dirname,'\axie_bump_atmsophere_inputs.txt'),'w');
    fprintf(fileID,'%8.6f\t',atm);
    fclose(fileID);
end

% Running Python command
if isempty(atm)
    if ~isempty(dirname)
        cd(strcat('.\',dirname));
    end
    system('python matlab_standard.py'); 
elseif length(atm)==7 % pull weather data in Python code from the given date/time
    if ~isempty(dirname)
        cd(strcat('.\',dirname));
    end
    system('python matlab_real_weather.py'); 
elseif length(atm)>=3 && length(atm)<=5 % generate weather data in Python code using autoencoder
    if ~isempty(dirname)
        cd(strcat('.\',dirname));
    end
    system('python matlab_real_weather_machine_learning.py');  % pwd needs to be in the folder with all machine learning stuff
%     system('python babysitter.py');  % pwd needs to be in the folder with all machine learning stuff
elseif atm % allow it to be a boolean true
    if ~isempty(dirname)
        cd(strcat('.\',dirname));
    end
    system('python matlab_from_input.py'); % use this command for when the input file is already written
else % use standard atmosphere
    warning('Check ''atm'' input. Standard Atmosphere is being used.')
    if ~isempty(dirname)
        cd(strcat('.\',dirname));
    end
    system('python matlab_standard.py'); 
end

% Reading output file
fileID = fopen('.\axie_bump_outputs.txt','r');
outputs = fscanf(fileID, '%f');
fclose(fileID);
perceived_loudness = outputs(1);

if ~isempty(dirname)
    cd ../
end
end
