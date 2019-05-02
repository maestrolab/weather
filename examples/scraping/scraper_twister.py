"""
Code developed to scrape weather data online from TwisterData.com.
 Data is scraped from the input Date and Time.
 Data is from the GFS weather model used by TwisterData.com.
"""

from weather.scraper.twister import scraper
import datetime

year = str(datetime.date.today().strftime('%Y'))
month = str(datetime.date.today().strftime('%m'))
day = str(datetime.date.today().strftime('%d'))
year = '2018'
month = '12'
day = '21'
hour = '12'

scraper(year, month, day, hour)
