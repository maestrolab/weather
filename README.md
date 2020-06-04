# weather
Library for obtaining data online or post-processing weather/boom data.

Weather data consist of GFS forecast or radiosonde measurements. For GFS, the data obtained is Pressure, Height, Temperature, Relative Humidity, Wind Speed, and Wind Direction at 4232 locations across the continental U.S. between -144 and -53 degrees West and 13 and 58 degrees North.


Installation:
-
  - open the command line in the main directory and run 'pip install -e .'.

Dependencies:
-
  - usuaero/rapidboom
  - usuaero/pyldb
  - bs4
  - scipy
  - matplotlib
  - mpl_toolkits

Utilization (check out examples):
- 
All script necessary to utilize the library are provided in examples and are broadly categorized as:
  - 'boom': constaints scripts to evaluate the loudness of sonic booms utilizing
      - balloon data: 'balloon_noise.py' for a single profile and 'balloon_noise_all.py' for all profiles in all station
      - NOAA GFS data: 'noaa/noise_all.py' for loudness of all Noth America, 'noaa/noise_POI.py' for loudness of a Point of Interest, and 'noaa/noise_interpolated_all.py' for loudness for specific locations for a whole year.
  - 'data_processing': contrains scripts to process
    - ADS-B data: 'airframe', ballon data 'balloon'
    - U.S. Census: 'census'
    - NOAA GFS: 'noaa/path_* ' for plots related to a specific flight path and 'noaa/noise_contour.py' for PLdB maps
    - weather profiles into clusters ('clustering.py')
  - 'humidity_parameterization': containts attempt in utilizing an autoencoder for parameterizing atmospheric profiles
  - 'scraping': constaints scripts to access online databases of:
    - ADS-B data: 'opensky_data.py'
    - Population: 'us_census.py'
    - Weather (Twister): 'scraper_twister.py'
    - Weather (radiosonde): 'weather_ballon.py'
    - Weather (NOAA): 'data_scraper' in 'matlab' directory
    

