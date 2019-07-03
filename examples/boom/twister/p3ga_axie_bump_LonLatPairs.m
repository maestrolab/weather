%% Description
% This script will iteratively call p3ga for each of the latitude and
% longitude pairs defined in lat_lon_pairs below. Note that the current
% code only allows integer latitude and longitude values.

lat_lon_pairs = [34 -118;
                35 -112;
                36 -105;
                37 -99;
                38 -93;
                39 -87;
                40 -80;
                41 -74];

%% Paths
% Add GitHub repos (Matlab codes)
    % Add p3ga repo folder and subfolders (this includes dd_tools)
    addpath(genpath('..\..\..\..\p3ga'))
    % Add randlc folder and subfolders
    addpath(genpath('..\..\..\..\randlc'))    

%% P3GA Problem Setup
nBumps = 1; % allow this many bumps
par = []; 
dom = 1; % PLdB
nvar = 3*nBumps; % height, length down the body, and width of the bump for each bump

% set up inequality constraints
A=zeros(nvar);
b=zeros(nvar,1);
counter=0;
for i=1:nBumps
    % the only constraint is that abs(h/w) < 0.02 for each bump
    j=(i-1)*3+1;
    counter=counter+1;
    A(counter,j:(j+2))=[1,0,-0.02];
    counter=counter+1;
    A(counter,j:(j+2))=[-1,0,-0.02];
end

% Constraints
Aeq=[];
beq=[];

% Bounds
lb = [-.05,0,0.125]; 
ub = [.05,32.92,2]; 
    % tile these bounds by the number of bumps
    lb=repmat(lb,1,nBumps);
    ub=repmat(ub,1,nBumps);
    

% Options
options = p3gaoptimset('q',4,...    svdd q parameter to use during search
	'qFront',10,...                 q parameter to use to find final non-dominated members
    'fracrej',0, ...                fracrej parameter for SVDD
    'MutationFraction', 0.05,...    mutation fraction of variables of an individual
    'CrossoverFraction',0.7,...     crossover fraction (changed from 0.8)
	'Generations',35,...            number of generations to evaluate
	'PopulationSize',100,...        size of the population at each generation
    'Dominance','search',...        algorithm for predictive dominance
    'Log',true,...                  log current data
    'Debug',true,...                log iSVDD data
    'GenerationData',false,...       save generation data (default is every 10 generations)
    'MaximumPopulation',inf,...    store only 5,000 population members at a time to improve speed later (should sacrifice some SVDD accuracy)
    'ViewProgress',false,...          whether or not to visualize the search (only works for 3 dimensions)
    'Vectorized',true,...           whether or not to run in parallel
    'InitFcn',{@initpop2}...         this initialization function uses a latin hypercube sampling to generate population members
			);

%% Setup Parallel process
if options.Vectorized
    pool=gcp('nocreate');
    if isempty(pool)
        pool=parpool;  % this line can be changed to load a specific parallel profile with a specific number of workers
    end

    addAttachedFiles(pool,{
        '.\',... % add current directory
...        '..\..\..\..\boom_opt\cases\AxisymmetricBump\axie_bump_run.m',... %add the matlab function
        })

    % set up worker directories:
    for i=1:pool.NumWorkers
        mkdir(strcat('dir_',num2str(pool.Cluster.Jobs.Tasks(i).Worker.ProcessId)));
        copyfile('matlab_real_weather.py', strcat('dir_',num2str(pool.Cluster.Jobs.Tasks(i).Worker.ProcessId)))
        addAttachedFiles(pool,{strcat('.\dir_',num2str(pool.Cluster.Jobs.Tasks(i).Worker.ProcessId))})
    end
end

%% Run P3GA
for i=1:size(lat_lon_pairs,1)
    FID = fopen('progress.txt','w');
    clc; fprintf(FID,'Working on %d of %d.',i,size(lat_lon_pairs,1));
    fclose(FID);
    
    tstart= tic;
    atm=[18,6,2018,12,lat_lon_pairs(i,1),lat_lon_pairs(i,2),50000]; % date, time, location to get atmospheric data
    fun = @(X)axie_bump_run(nBumps,X(1:3*nBumps),atm,true);
    [~, ~, M{i}] = p3ga(fun,dom,par,nvar,A,b,Aeq,beq,lb,ub,options);
    p3ga_elapsedtime{i}= toc(tstart)/3600; % elapsed time in hours
    save('current_solutions.mat')
end
    
% delete worker directories:
rmdir('dir_*', 's') % delete folders and all contents (* denotes all)