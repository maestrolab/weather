from setuptools import setup

setup(name='weather',
      version='0.1',
      description='Makes pictures from weather data',
      url='https://github.com/leal26/weather_model',
      author='Jake Schrass',
      author_email='jlostinco@tamu.edu',
      license='none',
      packages=['weather', 'weather.scrapper', 'weather.boom'],
      zip_safe=False)
