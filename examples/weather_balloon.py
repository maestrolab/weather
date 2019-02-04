#!python3
'''
Code that consolidates all aux needed to run any file in
 weather_module repository in alphabetical order.
'''
from weather.scraper import balloon_scraper
import pickle

YEAR = '2014'
MONTH = '01'
DAY = '07'
HOUR = '00'
data = balloon_scraper(YEAR, MONTH, DAY, HOUR)

g = open("file.p", "wb")
pickle.dump(data, g)
g.close()
