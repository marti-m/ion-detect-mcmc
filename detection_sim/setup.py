from setuptools import setup

setup(
   name='PyonDetect',
   version='0.1.0',
   description='A module used to simulate ion state detection using advanced ion dynamics and state discrimination algorithms',
   author='Michael Marti',
   author_email='micmarti@ethz.ch',
   packages=['PyonDetect'],  #same as name
   install_requires=['numpy', 'scipy', 'pandas'], #external packages as dependencies
)
