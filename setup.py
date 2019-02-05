from setuptools import setup

setup(name='weather',
      version='0.1',
      description='Makes pictures from weather data',
      url='https://github.com/leal26/weather_model',
      author='Pedro Leal',
      author_email='leal26@tamu.edu',
      license='none',
      packages=['weather', 'weather.scraper', 'weather.boom'],
      zip_safe=False)
