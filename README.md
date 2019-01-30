# weather
Library for obtaining data online from the GFS model from TwisterData.com. The data obtained is Pressure, Height, Temperature, Relative Humidity, Wind Speed, and Wind Direction at 4232 locations across the continental U.S. between -144 and -53 degrees West and 13 and 58 degrees North.


Installation:
  - open the command line in the main directory and run 'pip install -e .'.

Dependencies:
  - usuaero/rapidboom
  - usuaero/pyldb
  - bs4
  - scipy
  - matplotlib
  - mpl_toolkits

Utilization (check out examples):
  - Scrape data today from TwisterData (scraper_today.py)
  - Plot profiles for atmospheric data according to latitude (plot_profiles.py)
  - Propagate pressure signature and calculate ground level noise for a given latitude (scraper_today.py)

If you are from USU: do you ever icnorrectly write LDS and get LSD instead? I might
