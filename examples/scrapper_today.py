"""
Code developed to scrape weather data online from TwisterData.com.
 Data is scraped from the input Date and Time.
 Data is from the GFS weather model used by TwisterData.com.
"""

from weather.scrapper import scrapper
import datetime

year = str(datetime.date.today().strftime('%Y'))
month = str(datetime.date.today().strftime('%m'))
day = str(datetime.date.today().strftime('%d'))
hour = '12'

scrapper(year, month, day, hour)
